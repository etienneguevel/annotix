import pandas as pd
import torch
from torch import Tensor

from annotix_ml import BASE_DIR
from annotix_ml.graphtransf.data.dataset import GraphDatasetFromSMILEs
from annotix_ml.graphtransf.data.atoms_data import VALID_ELEMENTS, TYPE_EDGES
import rdkit.Chem as Chem


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
    assert len(nodes.size()) == 2
    assert len(edges.size()) == 3

    # Check that the nodes and edges are well one-hot encoded
    assert (nodes.sum(-1) == 1).all()
    assert (edges.sum(-1) == 1).all()


def test_dataset_distribution():
    # Open the MSG dataset
    df = pd.read_csv(BASE_DIR / "data" / "MassSpecGym.csv")

    # Select a part of the dataset
    df = df[df.fold == "train"].sample(100)
    dataset = GraphDatasetFromSMILEs(df, valid_elements=VALID_ELEMENTS)

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
    dataset = GraphDatasetFromSMILEs(df, valid_elements=VALID_ELEMENTS)

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
