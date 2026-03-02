import torch
import pandas as pd
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from annotix_ml import BASE_DIR
from annotix_ml.data.datacollator import collateGraph
from annotix_ml.data.dataset import GraphDatasetFromSMILEs
from annotix_ml.data.spec_dataset import GraphSpecDataset
from annotix_ml.data.atoms_data import TYPE_EDGES
from annotix_ml.spectraencoder.model.spectra_encoder import SpectraEncoderGrowing


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


def _make_spectra_encoder():
    cfg = _load_cfg()
    spectra_encoder = SpectraEncoderGrowing.init_from_cfg(cfg)

    return spectra_encoder


def test_collate_MSG():
    # Open the MSG dataset
    df = pd.read_csv(BASE_DIR / "data" / "MassSpecGym.csv")

    # Select a part of the dataset
    df = df[df.fold == "train"].sample(100)

    # Create the dataset & dataloader
    bs = 16
    dataset = GraphDatasetFromSMILEs(df)

    loader = DataLoader(dataset, batch_size=bs, collate_fn=collateGraph)

    N, E, node_mask = next(iter(loader))

    assert N.shape[0] == bs
    assert E.shape[0] == bs
    assert node_mask.shape[0] == bs
    assert len(N.shape) == 3
    assert len(E.shape) == 4


def test_collate_graph_spec():
    """Test collateGraphSpec produces correctly shaped and padded batches."""
    data_path = BASE_DIR / "data/BanyulsInference/labels.tsv"
    spec_folder = BASE_DIR / "data/BanyulsInference/spec_files"
    subform_folder = "data/BanyulsInference/subformulae/default_subformulae"

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.mps.is_available()
        else torch.device("cpu")
    )
    spectra_encoder = _make_spectra_encoder()
    spectra_encoder = spectra_encoder.to(device)

    dataset = GraphSpecDataset(
        data=data_path,
        spec_folder=spec_folder,
        subform_folder=subform_folder,
        smile_column="smiles",
        spec_column="spec",
        spectra_encoder=spectra_encoder,
        device=device,
        formula_column="formula",
        instrument_column="instrument",
    )

    bs = 4
    loader = DataLoader(dataset, batch_size=bs, collate_fn=collateGraph)
    (nodes, edges, mask, fp) = next(iter(loader))

    # Batch dimension
    assert nodes.shape[0] == bs
    assert edges.shape[0] == bs
    assert mask.shape[0] == bs
    assert fp.shape[0] == bs

    # Tensor ranks
    assert len(nodes.shape) == 3  # (bs, n, natoms)
    assert len(edges.shape) == 4  # (bs, n, n, nbonds)
    assert len(mask.shape) == 2  # (bs, n)

    # Spatial dimensions are consistent
    n = nodes.shape[1]
    assert edges.shape[1] == n
    assert edges.shape[2] == n
    assert mask.shape[1] == n

    # Edge bond dimension matches TYPE_EDGES
    assert edges.shape[-1] == len(TYPE_EDGES)
