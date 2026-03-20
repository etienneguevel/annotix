import os

import pandas as pd
import torch
from omegaconf import OmegaConf

from annotix_ml import BASE_DIR
from annotix_ml.graphtransf.arch.defog_meta_arch import DefogMetaArch
from annotix_ml.graphtransf.models.noising_defog import NoisingModelDefog
from annotix_ml.data.atoms_data import VALID_ELEMENTS, TYPE_EDGES
from annotix_ml.data.dataset import GraphDatasetFromSMILEs
from annotix_ml.graphtransf.test_utils import create_random_start


if os.path.exists(BASE_DIR / "data" / "MassSpecGym.csv"):
    df = pd.read_csv(BASE_DIR / "data" / "MassSpecGym.csv")
else:
    df = pd.read_csv(
        "hf://datasets/roman-bushuiev/MassSpecGym/data/MassSpecGym.tsv", sep="\t"
    )

df = df[df.fold == "train"].sample(1000)
train_dataset = GraphDatasetFromSMILEs(df)

_COMMON = dict(
    d=64,
    de=32,
    dy=16,
    n_heads=2,
    n_layers=2,
    diffusion_steps=5,
    loss_ratio=0.5,
    device=torch.device("cpu"),
    valid_elements=list(VALID_ELEMENTS),
    k=4,
    extra_features=["laplacian_embedding"],
    nodes_distribution=train_dataset.nodes_distribution,
    edges_distribution=train_dataset.edges_distribution,
)


def test_defog_meta_arch_initialization():
    arch = DefogMetaArch(noise_strategy="uniform", **_COMMON)

    assert isinstance(arch.noiser, NoisingModelDefog)
    assert arch.noiser.T == _COMMON["diffusion_steps"]
    assert arch.sample_steps == _COMMON["diffusion_steps"]
    assert arch.rdb == "general"
    assert arch.rdb_crit == "x_1"
    assert arch.eta == 0.0
    assert arch.omega == 0.0


def test_defog_transition_type_inferred_uniform():
    arch = DefogMetaArch(noise_strategy="uniform", **_COMMON)
    # limit_X should be uniform
    assert torch.allclose(
        arch.noiser.limit_X,
        torch.ones(len(VALID_ELEMENTS)) / len(VALID_ELEMENTS),
    )


def test_defog_transition_type_inferred_marginal():
    arch = DefogMetaArch(noise_strategy="marginal", **_COMMON)
    # limit_X should equal the normalised nodes_distribution
    nd = train_dataset.nodes_distribution.float()
    expected = nd / nd.sum()
    assert torch.allclose(arch.noiser.limit_X, expected, atol=1e-6)


def test_forward_backward():
    bs, n = 4, 15
    arch = DefogMetaArch(noise_strategy="uniform", **_COMMON)

    N, E, mask = create_random_start(bs, n, len(TYPE_EDGES), len(VALID_ELEMENTS))
    pN, pE, loss = arch.forward_backward(N, E, mask)

    assert pN is not None
    assert pE is not None
    assert loss is not None
    assert pN.shape == (bs, n, len(VALID_ELEMENTS))
    assert pE.shape == (bs, n, n, len(TYPE_EDGES))
    assert loss.ndim == 0


def test_generate_shapes():
    arch = DefogMetaArch(noise_strategy="uniform", **_COMMON)

    batch_size = 4
    max_nodes = 9
    N, E, mask = arch.generate(batch_size, max_nodes)

    assert N.shape == (batch_size, max_nodes, len(VALID_ELEMENTS))
    assert E.shape == (batch_size, max_nodes, max_nodes, len(TYPE_EDGES))
    assert mask.shape == (batch_size, max_nodes)


def test_generate_one_hot():
    arch = DefogMetaArch(noise_strategy="uniform", **_COMMON)

    batch_size = 4
    max_nodes = 9
    N, E, mask = arch.generate(batch_size, max_nodes)

    assert (N >= 0).all() and (N <= 1).all()
    assert (E >= 0).all() and (E <= 1).all()
    assert torch.allclose(N.sum(-1), torch.ones(batch_size, max_nodes), atol=1e-5)
    assert torch.allclose(
        E.sum(-1), torch.ones(batch_size, max_nodes, max_nodes), atol=1e-5
    )


def test_generate_with_fixed_sizes():
    arch = DefogMetaArch(noise_strategy="uniform", **_COMMON)

    n_sizes = torch.tensor([6, 7, 8, 9])
    max_nodes = 9
    N, E, mask = arch.generate(n_sizes, max_nodes)

    assert N.shape[0] == 4
    assert mask.shape == (4, max_nodes)


def test_compute_step_probs_row_sum():
    bs, n, natoms = 2, 5, len(VALID_ELEMENTS)
    nbonds = len(TYPE_EDGES)

    R_t_X = torch.rand(bs, n, natoms) * 0.1
    R_t_E = torch.rand(bs, n, n, nbonds) * 0.1
    X_t = torch.zeros(bs, n, natoms)
    X_t.scatter_(-1, torch.randint(0, natoms, (bs, n, 1)), 1.0)
    E_t = torch.zeros(bs, n, n, nbonds)
    E_t.scatter_(-1, torch.randint(0, nbonds, (bs, n, n, 1)), 1.0)
    dt = 0.1

    sp_X, sp_E = DefogMetaArch._compute_step_probs(R_t_X, R_t_E, X_t, E_t, dt)

    assert torch.allclose(sp_X.sum(-1), torch.ones(bs, n), atol=1e-5)
    assert torch.allclose(sp_E.sum(-1), torch.ones(bs, n, n), atol=1e-5)


def test_init_from_cfg():
    cfg = OmegaConf.load(BASE_DIR / "configs" / "default_config.yaml")
    cfg.model.noise_strategy = "marginal"
    arch = DefogMetaArch.init_from_cfg(
        cfg,
        torch.device("cpu"),
        list(VALID_ELEMENTS),
        train_dataset.nodes_distribution,
        train_dataset.edges_distribution,
    )
    assert isinstance(arch, DefogMetaArch)
    assert isinstance(arch.noiser, NoisingModelDefog)
