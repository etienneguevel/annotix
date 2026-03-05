import os

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from annotix_ml import BASE_DIR
from annotix_ml.data.atoms_data import VALID_ELEMENTS, TYPE_EDGES
from annotix_ml.data.dataset import GraphDatasetFromSMILEs
from annotix_ml.graphtransf.arch.digress_meta_arch import DigressMetaArch
from annotix_ml.train.eval import generate_samples_from_fp


# Module-level dataset + small model setup (shared across all tests for speed)
if os.path.exists(BASE_DIR / "data" / "MassSpecGym.csv"):
    df = pd.read_csv(BASE_DIR / "data" / "MassSpecGym.csv")
else:
    df = pd.read_csv(
        "hf://datasets/roman-bushuiev/MassSpecGym/data/MassSpecGym.tsv", sep="\t"
    )

df = df[df.fold == "train"].sample(1000)
train_dataset = GraphDatasetFromSMILEs(df)

_N_MAX = 10
_BATCH_SIZE = 3
_DEVICE = torch.device("cpu")

_model = DigressMetaArch(
    d=64,
    de=32,
    dy=16,
    n_heads=2,
    n_layers=2,
    noise_strategy="uniform",
    diffusion_steps=5,
    loss_ratio=0.5,
    device=_DEVICE,
    valid_elements=list(VALID_ELEMENTS),
    k=4,
    extra_features=["laplacian_embedding", "spectra_fingerprint"],
    nodes_distribution=train_dataset.nodes_distribution,
    edges_distribution=train_dataset.edges_distribution,
)


def _make_dataloader(n_batches: int, batch_size: int) -> DataLoader:
    """Build a DataLoader yielding (nodes, edges, mask, fp) with float32 mask.

    Creates tensors with explicit num_classes so shapes are consistent across batches.
    mask is float32 (torch.ones/zeros), matching the dtype produced in production by
    collateGraph — this is the condition that triggers the bug under test.
    """
    n_atoms = len(VALID_ELEMENTS)
    n_bonds = len(TYPE_EDGES)
    total = n_batches * batch_size

    nodes_idx = torch.randint(high=n_atoms, size=(total, _N_MAX))
    edges_idx = torch.randint(high=n_bonds, size=(total, _N_MAX, _N_MAX))
    N = F.one_hot(nodes_idx, num_classes=n_atoms).float()  # (total, n_max, n_atoms)
    E = F.one_hot(
        edges_idx, num_classes=n_bonds
    ).float()  # (total, n_max, n_max, n_bonds)

    # Build float32 mask (low=1 ensures at least one valid node per sample)
    counts = torch.randint(low=1, high=_N_MAX, size=(total,))
    mask = torch.stack(
        [torch.cat([torch.ones(int(m)), torch.zeros(_N_MAX - int(m))]) for m in counts]
    )  # (total, n_max), dtype=float32

    N = N * mask.unsqueeze(-1)
    E = E * mask.unsqueeze(-1).unsqueeze(-1)

    fp = torch.rand((total, _model.morgan_nbits))

    dataset = TensorDataset(N, E, mask, fp)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def test_generate_samples_from_fp_returns_expected_keys():
    loader = _make_dataloader(n_batches=2, batch_size=_BATCH_SIZE)
    result = generate_samples_from_fp(
        _model, num_samples=_BATCH_SIZE, eval_dataloader=loader
    )

    expected_keys = {
        "all_gen_smiles",
        "all_gen_smiles_digress",
        "true_smiles",
        "validity",
        "validity_digress",
        "tan_sim",
        "mces",
    }
    assert set(result.keys()) == expected_keys


def test_generate_samples_from_fp_list_lengths_match():
    loader = _make_dataloader(n_batches=2, batch_size=_BATCH_SIZE)
    result = generate_samples_from_fp(
        _model, num_samples=_BATCH_SIZE, eval_dataloader=loader
    )

    lengths = {k: len(v) for k, v in result.items()}
    unique_lengths = set(lengths.values())
    assert len(unique_lengths) == 1, f"List lengths differ: {lengths}"


def test_generate_samples_from_fp_float_mask_no_crash():
    """Regression: float32 mask must not cause TypeError inside model.generate()."""
    loader = _make_dataloader(n_batches=2, batch_size=_BATCH_SIZE)

    # Verify the mask dtype is float32 — this is what triggers the bug before the fix
    _, _, mask, _ = next(iter(loader))
    assert mask.dtype == torch.float32

    # With the fix (int cast), this must not raise TypeError
    result = generate_samples_from_fp(
        _model, num_samples=_BATCH_SIZE, eval_dataloader=loader
    )
    assert "all_gen_smiles" in result


def test_generate_samples_from_fp_stops_at_num_samples():
    """Function stops after accumulating num_samples, even if the loader has more batches."""
    loader = _make_dataloader(n_batches=4, batch_size=_BATCH_SIZE)
    result = generate_samples_from_fp(
        _model, num_samples=_BATCH_SIZE, eval_dataloader=loader
    )

    # Only one batch should have been processed
    assert len(result["all_gen_smiles"]) == _BATCH_SIZE


def test_generate_samples_from_fp_exhausted_loader():
    """Function exits gracefully when the dataloader runs out before num_samples."""
    loader = _make_dataloader(n_batches=1, batch_size=_BATCH_SIZE)
    result = generate_samples_from_fp(_model, num_samples=100, eval_dataloader=loader)

    # Only one batch worth of samples should be present
    assert len(result["all_gen_smiles"]) == _BATCH_SIZE
