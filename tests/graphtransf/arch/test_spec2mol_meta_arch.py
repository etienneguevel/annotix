import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from annotix_ml import BASE_DIR
from annotix_ml.graphtransf.arch.spec2mol_meta_arch import Spec2MolMetaArch
from annotix_ml.data.spec_dataset import GraphSpecDataset
from annotix_ml.data.atoms_data import TYPE_EDGES
from annotix_ml.data.datacollator import collateGraphSpec
from annotix_ml.data.data_utils import graph_to_smiles_digress


def _load_cfg():
    # Load default config
    default_cfg_path = BASE_DIR / "configs" / "default_config.yaml"
    default_cfg = OmegaConf.load(default_cfg_path)

    # Load the spec2mol config
    spectra_cfg = OmegaConf.load(
        BASE_DIR / "configs" / "spec2mol" / "spectral_feat" / "baseline.yaml"
    )

    # Merge the configs
    cfg = OmegaConf.merge(default_cfg, spectra_cfg)

    if "spectra_fingerprint" not in cfg.model.extra_features:
        cfg.model.extra_features.append("spectra_fingerprint")
    return cfg


def _make_dataset(cfg):
    data = pd.read_csv(
        cfg.dataset.data_path,
        sep="\t" if str(cfg.dataset.data_path).endswith(".tsv") else ",",
    )

    split_column = cfg.dataset.split_column
    val_tag = cfg.dataset.val_tag

    train_df = data[data[split_column] != val_tag].reset_index(drop=True).sample(1000)

    max_nodes = cfg.dataset.get("max_nodes")
    dataset = GraphSpecDataset(
        data=train_df,
        save_cache=False,
        spec_folder=cfg.dataset.spec_folder,
        subform_folder=cfg.dataset.subform_folder,
        smile_column=cfg.dataset.smile_column,
        sanitizer=graph_to_smiles_digress,
        verbose=True,
        max_nodes=max_nodes,
    )

    return dataset


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
    loader = DataLoader(dataset, batch_size=bs, collate_fn=collateGraphSpec)
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
    )


def test_spec2mol_meta_arch_initialization():
    """Test basic initialization of Spec2MolMetaArch."""
    cfg = _load_cfg()

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
    if not cfg.model.no_y:
        assert model.diffuser.dy == cfg.model.dy


def test_compute_extra_features():
    """Test compute_extra_features includes spectra fingerprint in global features."""
    cfg = _load_cfg()
    dataset = _make_dataset(cfg)
    model = _make_model(cfg, dataset)
    batch = _load_batch(dataset)

    bs = batch["nodes"].shape[0]
    t = torch.randint(1, cfg.model.diffusion_steps, (bs, 1))

    args = (
        batch["num_peaks"],
        batch["types"],
        batch["instruments"],
        batch["ion_vec"],
        batch["form_vec"],
        batch["intens"],
    )

    pos_emb, y = model.compute_extra_features(
        batch["nodes"], batch["edges"], batch["mask"], t, *args
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

    args = (
        batch["num_peaks"],
        batch["types"],
        batch["instruments"],
        batch["ion_vec"],
        batch["form_vec"],
        batch["intens"],
    )

    pN, pE = model.forward(batch["nodes"], batch["edges"], batch["mask"], t, *args)

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

    args = (
        batch["num_peaks"],
        batch["types"],
        batch["instruments"],
        batch["ion_vec"],
        batch["form_vec"],
        batch["intens"],
    )

    model.spectra_encoder.train()
    pN, pE, loss = model.forward_backward(
        batch["nodes"], batch["edges"], batch["mask"], *args
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
    args = (
        batch["num_peaks"][:batch_size],
        batch["types"][:batch_size],
        batch["instruments"][:batch_size],
        batch["ion_vec"][:batch_size],
        batch["form_vec"][:batch_size],
        batch["intens"][:batch_size],
    )

    N, E, gen_mask = model.generate(batch_size, max_nodes, other_args=args)

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
