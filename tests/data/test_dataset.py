import pandas as pd
import rdkit.Chem as Chem
import torch
from torch import Tensor

from annotix_ml import BASE_DIR
from annotix_ml.data.dataset import GraphDatasetFromSMILEs, GraphDatasetWithFP
from annotix_ml.data.atoms_data import VALID_ELEMENTS, TYPE_EDGES


def test_dataset_msg():
    # Open the MSG dataset
    df = pd.read_csv(BASE_DIR / "data" / "MassSpecGym.csv")

    # Select a part of the dataset
    df = df[df.fold == "train"].sample(100)

    # Create the dataset
    dataset = GraphDatasetFromSMILEs(df)

    # Check if different methodes are as expected
    assert isinstance(len(dataset), int)

    nodes, edges = dataset[0]
    assert isinstance(nodes, Tensor)
    assert isinstance(edges, Tensor)

    # Check the correct unpacking
    assert nodes.dim() == 2
    assert edges.dim() == 3

    # Check that the nodes and edges are well one-hot encoded
    assert (nodes.sum(-1) == 1).all()
    assert (edges.sum(-1) == 1).all()


def test_dataset_distribution():
    # Open the MSG dataset
    df = pd.read_csv(BASE_DIR / "data" / "MassSpecGym.csv")

    # Select a part of the dataset
    df = df[df.fold == "train"].sample(100)
    dataset = GraphDatasetFromSMILEs(df, valid_elements=list(VALID_ELEMENTS))

    # Check the distribution sizes
    node_distribution = dataset.nodes_distribution
    edge_distribution = dataset.edges_distribution

    assert node_distribution.shape == (len(VALID_ELEMENTS),)
    assert edge_distribution.shape == (len(TYPE_EDGES),)

    # Check sums are approximately 1 (use a small tolerance)
    assert torch.allclose(node_distribution.sum(), torch.tensor(1.0), atol=1e-4)
    assert torch.allclose(edge_distribution.sum(), torch.tensor(1.0), atol=1e-4)


def test_graph_to_smiles():
    # Open the MSG dataset
    df = pd.read_csv(BASE_DIR / "data" / "MassSpecGym.csv")

    # Select a part of the dataset
    df = df[df.fold == "train"].sample(100)
    dataset = GraphDatasetFromSMILEs(df)

    # List of smiles to test
    smiles_list = ["C", "CC", "CCO", "c1ccccc1", "C1CCCCC1", "C(=O)O"]

    for sm in smiles_list:
        # Convert to graph
        nodes, edges = dataset.smilesToGraph(sm)

        # Convert back to smiles
        reconstructed_smiles = dataset.graphToSmiles(nodes, edges)

        # Check if the smiles are the same
        # We canonicalize both just in case
        original_mol = Chem.MolFromSmiles(sm)
        reconstructed_mol = Chem.MolFromSmiles(reconstructed_smiles)

        assert original_mol is not None
        assert reconstructed_mol is not None

        original_canon = Chem.MolToSmiles(original_mol, canonical=True)
        reconstructed_canon = Chem.MolToSmiles(reconstructed_mol, canonical=True)

        assert original_canon == reconstructed_canon, (
            f"Failed for {sm}: {original_canon} != {reconstructed_canon}"
        )


def test_graph_dataset_withfp():
    # Open the MSG dataset
    df = pd.read_csv(BASE_DIR / "data" / "MassSpecGym.csv")

    # Select a part of the dataset
    df = df[df.fold == "train"].sample(100)

    # Create the dataset
    dataset = GraphDatasetWithFP(df)

    # Test the elements in the dataset
    nodes, edges, fp = dataset[0]
    assert nodes.dim() == 2
    assert edges.dim() == 3
    assert fp.dim() == 1


def _split_df_missing_element_in_val():
    # Train contains Cl, val does not -> val auto-detection would yield fewer elements
    return pd.DataFrame(
        {
            "smiles": ["CCCl", "CCO", "CCN", "CCC", "CCO", "CN"],
            "fold": ["train", "train", "train", "test", "test", "test"],
        }
    )


def test_make_datasets_val_uses_train_elements(tmp_path):
    from annotix_ml.data.loaders import make_datasets

    data_path = tmp_path / "data.csv"
    _split_df_missing_element_in_val().to_csv(data_path, index=False)

    train_ds, valid_ds = make_datasets(
        str(data_path), split_column="fold", val_tag="test", verbose=False
    )

    assert "Cl" in train_ds.valid_elements
    assert valid_ds.valid_elements == train_ds.valid_elements
    nodes, _ = valid_ds[0]
    assert nodes.shape[-1] == len(train_ds.valid_elements)


def test_cache_rebuilt_on_valid_elements_mismatch(tmp_path):
    df = _split_df_missing_element_in_val()
    val_df = df[df.fold == "test"]
    cache_path = str(tmp_path / "cache_valid.pt")

    # Build a stale cache with auto-detected (reduced) elements
    stale = GraphDatasetFromSMILEs(val_df, cache_path=cache_path, verbose=False)
    assert "Cl" not in stale.valid_elements

    expected = ["C", "Cl", "N", "O"]
    ds = GraphDatasetFromSMILEs(
        val_df, valid_elements=expected, cache_path=cache_path, verbose=False
    )
    assert ds.valid_elements == expected
    assert ds[0][0].shape[-1] == len(expected)
    assert torch.load(cache_path, weights_only=False)["valid_elements"] == expected
