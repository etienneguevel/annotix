import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from annotix_ml import BASE_DIR
from annotix_ml.spec2mol.spec2mol_meta_arch import Spec2MolMetaArch
from annotix_ml.spec2mol.data.dataset import GraphSpecDataset
from annotix_ml.spec2mol.data.datacollator import graph_spec_collate_fn
from annotix_ml.graphtransf.data.atoms_data import TYPE_EDGES


def test_spec2mol_meta_arch_initialization():
    """Test basic initialization of Spec2MolMetaArch"""
    cfg = OmegaConf.load(BASE_DIR / "configs" / "default_config.yaml")

    # Mock elements and distributions
    valid_elements = ["C", "H", "O", "N"]
    nodes_distribution = torch.ones(len(valid_elements)) / len(valid_elements)
    edges_distribution = torch.ones(len(TYPE_EDGES)) / len(TYPE_EDGES)
    device = torch.device("cpu")

    model = Spec2MolMetaArch.init_from_cfg(
        cfg=cfg,
        device=device,
        valid_elements=valid_elements,
        nodes_distribution=nodes_distribution,
        edges_distribution=edges_distribution,
        max_weight=500.0,
    )

    assert model is not None
    assert isinstance(model.spectra_encoder, torch.nn.Module)
    assert model.device == device


def test_spec2mol_meta_arch_forward_backward():
    """Test compute_extra_features, forward, and forward_backward using real data"""
    # Load config
    cfg = OmegaConf.load(BASE_DIR / "configs" / "default_config.yaml")

    if "spectra_fingerprint" not in cfg.model.extra_features:
        cfg.model.extra_features.append("spectra_fingerprint")

    # Use real dataset paths from config
    data_path = BASE_DIR / cfg.dataset.labels_file
    spec_folder = BASE_DIR / cfg.dataset.spec_folder
    subform_folder = BASE_DIR / cfg.dataset.subform_folder

    # Create dataset
    dataset = GraphSpecDataset(
        data=data_path,
        spec_folder=spec_folder,
        subform_folder=subform_folder,
        smile_column="smiles",
        spec_column="spec",
        formula_column="formula",
        instrument_column="instrument",
    )

    # Collate
    loader = DataLoader(dataset, batch_size=2, collate_fn=graph_spec_collate_fn)
    (
        nodes,
        edges,
        mask,
        num_peaks,
        types,
        instruments,
        ion_vec,
        form_vec,
        intens,
        smiles,
    ) = next(iter(loader))
    nodes = nodes.float()
    edges = edges.float()

    # Init model
    device = torch.device("cpu")
    model = Spec2MolMetaArch.init_from_cfg(
        cfg=cfg,
        device=device,
        valid_elements=dataset.valid_elements,
        nodes_distribution=dataset.nodes_distribution,
        edges_distribution=dataset.edges_distribution,
        max_weight=dataset.max_weight,
    )

    # 1. Test compute_extra_features
    # We need a timestep 't' in the batch for compute_extra_features
    t = torch.randint(1, cfg.model.diffusion_steps, (nodes.shape[0], 1))

    # Ensure all required spectra keys are in batch
    pos_emb, y = model.compute_extra_features(
        nodes=nodes,
        edges=edges,
        mask=mask,
        t=t,
        num_peaks=num_peaks,
        types=types,
        instruments=instruments,
        ion_vec=ion_vec,
        form_vec=form_vec,
        intens=intens,
    )

    assert pos_emb.shape[0] == 2
    assert y.shape[0] == 2
    assert y.shape[-1] == model.global_features

    # 2. Test forward
    # Forward requires "nodes", "edges", "mask"
    # graph_spec_collate_fn returns "mask", but DigressMetaArch expects "mask"
    # Wait, let's check DigressMetaArch.forward signature

    # Actually, graph_spec_collate_fn returns "mask" (torch.bool)
    # DigressMetaArch might expect "mask" or something else.
    # Looking at digress_meta_arch.py, it uses batch["mask"]

    p_n, p_e = model.forward(
        nodes=nodes,
        edges=edges,
        mask=mask,
        num_peaks=num_peaks,
        types=types,
        instruments=instruments,
        ion_vec=ion_vec,
        form_vec=form_vec,
        intens=intens,
    )

    assert p_n.shape[0] == 2
    assert p_e.shape[0] == 2
    assert p_n.shape[-1] == len(dataset.valid_elements)
    assert p_e.shape[-1] == len(TYPE_EDGES)

    # 3. Test compute_loss
    # Reconstruct a batch dictionary for compute_loss which still expects it
    batch_dict = {"nodes": nodes, "edges": edges, "mask": mask}

    outputs = model.forward(
        nodes=nodes,
        edges=edges,
        mask=mask,
        num_peaks=num_peaks,
        types=types,
        instruments=instruments,
        ion_vec=ion_vec,
        form_vec=form_vec,
        intens=intens,
    )
    loss, metrics = model.compute_loss(batch_dict, outputs)

    assert loss is not None
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert "node_accuracy" in metrics
    assert "edge_accuracy" in metrics
    for at in model.valid_elements:
        assert f"ce_{at}" in metrics, f"Missing cross-entropy metric for atom {at}"
