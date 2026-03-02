import pandas as pd
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from annotix_ml import BASE_DIR
from annotix_ml.graphtransf.arch.spec2mol_meta_arch import Spec2MolMetaArch
from annotix_ml.data.spec_dataset import GraphSpecDataset
from annotix_ml.data.atoms_data import TYPE_EDGES
from annotix_ml.data.datacollator import collateGraph
from annotix_ml.data.data_utils import graph_to_smiles_digress
from annotix_ml.spectraencoder.model.spectra_encoder import SpectraEncoderGrowing


DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.mps.is_available()
    else torch.device("cpu")
)


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

    # Init the spectra_encoder
    spectra_encoder = SpectraEncoderGrowing.init_from_cfg(cfg)

    # Move to the local device
    spectra_encoder = spectra_encoder.to(DEVICE)

    # Create the dataset
    max_nodes = cfg.dataset.get("max_nodes")
    dataset = GraphSpecDataset(
        data=train_df,
        save_cache=False,
        spec_folder=cfg.dataset.spec_folder,
        subform_folder=cfg.dataset.subform_folder,
        smile_column=cfg.dataset.smile_column,
        spectra_encoder=spectra_encoder,
        device=DEVICE,
        sanitizer=graph_to_smiles_digress,
        verbose=True,
        max_nodes=max_nodes,
    )

    return dataset


def _make_model(cfg, dataset):
    # Move to the local device
    return Spec2MolMetaArch.init_from_cfg(
        cfg=cfg,
        device=DEVICE,
        valid_elements=dataset.valid_elements,
        nodes_distribution=dataset.nodes_distribution,
        edges_distribution=dataset.edges_distribution,
        max_weight=dataset.max_weight,
    )


def _load_batch(dataset, bs=2):
    loader = DataLoader(dataset, batch_size=bs, collate_fn=collateGraph)
    batch = (i.to(dataset.device) for i in next(iter(loader)))

    return batch


def test_spec2mol_meta_arch_initialization():
    """Test basic initialization of Spec2MolMetaArch."""
    cfg = _load_cfg()

    valid_elements = ["C", "H", "O", "N"]
    nodes_distribution = torch.ones(len(valid_elements)) / len(valid_elements)
    edges_distribution = torch.ones(len(TYPE_EDGES)) / len(TYPE_EDGES)

    model = Spec2MolMetaArch.init_from_cfg(
        cfg=cfg,
        device=DEVICE,
        valid_elements=valid_elements,
        nodes_distribution=nodes_distribution,
        edges_distribution=edges_distribution,
        max_weight=500.0,
    )

    assert model is not None
    assert model.device == DEVICE


def test_spec2mol_meta_arch_initialization_from_config():
    """Test initialization from config with real dataset distributions."""
    cfg = _load_cfg()
    dataset = _make_dataset(cfg)
    model = _make_model(cfg, dataset)

    assert model.loss_ratio == cfg.train.loss_ratio
    assert model.device == DEVICE
    assert model.diffuser is not None
    assert model.noiser is not None
    if not cfg.model.no_y:
        assert model.diffuser.dy == cfg.model.dy

    assert isinstance(dataset.spectra_encoder, nn.Module)


def test_compute_extra_features():
    """Test compute_extra_features includes spectra fingerprint in global features."""
    cfg = _load_cfg()
    dataset = _make_dataset(cfg)
    model = _make_model(cfg, dataset)
    nodes, edges, mask, fingerprints = _load_batch(dataset)

    bs = nodes.shape[0]
    t = torch.randint(1, cfg.model.diffusion_steps, (bs, 1))

    pos_emb, y = model.compute_extra_features(nodes, edges, mask, t)

    assert pos_emb.shape[0] == bs
    assert pos_emb.shape[-1] == model.node_features
    assert y.shape[0] == bs
    assert (y.shape[-1] + fingerprints.shape[-1]) == model.global_features


def test_forward():
    """Test the forward pass produces correct output shapes."""
    cfg = _load_cfg()
    dataset = _make_dataset(cfg)
    model = _make_model(cfg, dataset)
    nodes, edges, mask, fingerprints = _load_batch(dataset)

    bs = nodes.shape[0]
    t = torch.randint(1, cfg.model.diffusion_steps, (bs, 1))

    pN, pE = model.forward(nodes, edges, mask, t, fingerprints)

    assert pN.shape[0] == bs
    assert pE.shape[0] == bs
    assert pN.shape[-1] == len(dataset.valid_elements)
    assert pE.shape[-1] == len(TYPE_EDGES)


def test_forward_backward():
    """Test forward_backward computes loss and backpropagates."""
    cfg = _load_cfg()
    dataset = _make_dataset(cfg)
    model = _make_model(cfg, dataset)
    nodes, edges, mask, fingerprints = _load_batch(dataset)

    bs = nodes.shape[0]

    model.diffuser.train()
    pN, pE, loss = model.forward_backward(nodes, edges, mask, fingerprints)

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
    nodes, edges, _, fingerprints = _load_batch(dataset)

    batch_size = 2
    max_nodes = 9

    N, E, gen_mask = model.generate(batch_size, max_nodes, fp=fingerprints)

    # Verify output shapes
    assert N.shape[0] == batch_size
    assert E.shape[0] == batch_size
    assert N.shape[-1] == len(dataset.valid_elements)
    assert E.shape[-1] == len(TYPE_EDGES)

    n = N.shape[1]

    # Verify outputs are one-hot encoded
    assert torch.allclose(
        N.sum(dim=-1).float(), torch.ones(batch_size, n, device=DEVICE), atol=1e-5
    )
    assert torch.allclose(
        E.sum(dim=-1).float(), torch.ones(batch_size, n, n, device=DEVICE), atol=1e-5
    )

    # Verify outputs are valid (all values between 0 and 1)
    assert (N >= 0).all() and (N <= 1).all()
    assert (E >= 0).all() and (E <= 1).all()
