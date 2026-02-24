import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from annotix_ml import BASE_DIR
from annotix_ml.graphtransf.arch.spec2mol_meta_arch import Spec2MolMetaArch
from annotix_ml.data.spec_dataset import GraphSpecDataset
from annotix_ml.data.datacollator import graph_spec_collate_fn
from annotix_ml.data.atoms_data import TYPE_EDGES


def _load_cfg():
    cfg = OmegaConf.load(BASE_DIR / "configs" / "default_config.yaml")
    if "spectra_fingerprint" not in cfg.model.extra_features:
        cfg.model.extra_features.append("spectra_fingerprint")
    return cfg


def _make_dataset(cfg):
    return GraphSpecDataset(
        data=BASE_DIR / cfg.dataset.labels_file,
        spec_folder=BASE_DIR / cfg.dataset.spec_folder,
        subform_folder=cfg.dataset.subform_folder,
        smile_column="smiles",
        spec_column="spec",
        formula_column="formula",
        instrument_column="instrument",
    )


def _make_model(cfg, dataset):
    return Spec2MolMetaArch.init_from_cfg(
        cfg=cfg,
        device=torch.device("cpu"),
        valid_elements=dataset.valid_elements,
        nodes_distribution=dataset.nodes_distribution,
        edges_distribution=dataset.edges_distribution,
        max_weight=dataset.max_weight,
    )


def _load_batch(dataset, bs=2):
    loader = DataLoader(dataset, batch_size=bs, collate_fn=graph_spec_collate_fn)
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
    return dict(
        nodes=nodes.float(),
        edges=edges.float(),
        mask=mask,
        num_peaks=num_peaks,
        types=types,
        instruments=instruments,
        ion_vec=ion_vec,
        form_vec=form_vec,
        intens=intens,
        smiles=smiles,
    )


def test_spec2mol_meta_arch_initialization():
    """Test basic initialization of Spec2MolMetaArch."""
    cfg = OmegaConf.load(BASE_DIR / "configs" / "default_config.yaml")

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


def test_spec2mol_meta_arch_initialization_from_config():
    """Test initialization from config with real dataset distributions."""
    cfg = _load_cfg()
    dataset = _make_dataset(cfg)
    model = _make_model(cfg, dataset)

    assert model.loss_ratio == cfg.train.loss_ratio
    assert model.device == torch.device("cpu")
    assert model.diffuser is not None
    assert model.noiser is not None
    assert model.spectra_encoder is not None
    assert model.merge_function is not None
    assert model.diffuser.dy == cfg.model.dy


def test_compute_extra_features():
    """Test compute_extra_features includes spectra fingerprint in global features."""
    cfg = _load_cfg()
    dataset = _make_dataset(cfg)
    model = _make_model(cfg, dataset)
    batch = _load_batch(dataset)

    bs = batch["nodes"].shape[0]
    t = torch.randint(1, cfg.model.diffusion_steps, (bs, 1))

    pos_emb, y = model.compute_extra_features(
        nodes=batch["nodes"],
        edges=batch["edges"],
        mask=batch["mask"],
        t=t,
        num_peaks=batch["num_peaks"],
        types=batch["types"],
        instruments=batch["instruments"],
        ion_vec=batch["ion_vec"],
        form_vec=batch["form_vec"],
        intens=batch["intens"],
    )

    assert pos_emb.shape[0] == bs
    assert pos_emb.shape[-1] == model.node_features
    assert y.shape[0] == bs
    assert y.shape[-1] == model.global_features


def test_forward():
    """Test the forward pass produces correct output shapes."""
    cfg = _load_cfg()
    dataset = _make_dataset(cfg)
    model = _make_model(cfg, dataset)
    batch = _load_batch(dataset)

    bs = batch["nodes"].shape[0]
    t = torch.randint(1, cfg.model.diffusion_steps, (bs, 1))

    pN, pE = model.forward(
        nodes=batch["nodes"],
        edges=batch["edges"],
        mask=batch["mask"],
        t=t,
        num_peaks=batch["num_peaks"],
        types=batch["types"],
        instruments=batch["instruments"],
        ion_vec=batch["ion_vec"],
        form_vec=batch["form_vec"],
        intens=batch["intens"],
    )

    assert pN.shape[0] == bs
    assert pE.shape[0] == bs
    assert pN.shape[-1] == len(dataset.valid_elements)
    assert pE.shape[-1] == len(TYPE_EDGES)


def test_forward_backward():
    """Test forward_backward computes loss and backpropagates."""
    cfg = _load_cfg()
    dataset = _make_dataset(cfg)
    model = _make_model(cfg, dataset)
    batch = _load_batch(dataset)

    bs = batch["nodes"].shape[0]

    model.spectra_encoder.train()
    pN, pE, loss = model.forward_backward(
        nodes=batch["nodes"],
        edges=batch["edges"],
        mask=batch["mask"],
        num_peaks=batch["num_peaks"],
        types=batch["types"],
        instruments=batch["instruments"],
        ion_vec=batch["ion_vec"],
        form_vec=batch["form_vec"],
        intens=batch["intens"],
    )

    assert pN is not None
    assert pE is not None
    assert loss is not None
    assert pN.shape[0] == bs
    assert pE.shape[0] == bs
    assert pN.shape[-1] == len(dataset.valid_elements)
    assert pE.shape[-1] == len(TYPE_EDGES)
    assert loss.dim() == 0


def test_generate():
    """Test molecule generation conditioned on spectra."""
    cfg = _load_cfg()
    dataset = _make_dataset(cfg)
    model = _make_model(cfg, dataset)
    batch = _load_batch(dataset)

    batch_size = 2
    max_nodes = 9

    # Generate with spectral conditioning
    N, E, gen_mask = model.generate(
        num_samples=batch_size,
        max_nodes=max_nodes,
        num_peaks=batch["num_peaks"][:batch_size],
        types=batch["types"][:batch_size],
        instruments=batch["instruments"][:batch_size],
        ion_vec=batch["ion_vec"][:batch_size],
        form_vec=batch["form_vec"][:batch_size],
        intens=batch["intens"][:batch_size],
    )

    # Verify output shapes
    assert N.shape[0] == batch_size
    assert E.shape[0] == batch_size
    assert N.shape[-1] == len(dataset.valid_elements)
    assert E.shape[-1] == len(TYPE_EDGES)

    n = N.shape[1]

    # Verify outputs are one-hot encoded
    assert torch.allclose(N.sum(dim=-1).float(), torch.ones(batch_size, n), atol=1e-5)
    assert torch.allclose(
        E.sum(dim=-1).float(), torch.ones(batch_size, n, n), atol=1e-5
    )

    # Verify outputs are valid (all values between 0 and 1)
    assert (N >= 0).all() and (N <= 1).all()
    assert (E >= 0).all() and (E <= 1).all()
