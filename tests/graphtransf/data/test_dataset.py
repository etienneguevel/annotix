import pandas as pd
import torch
from torch import Tensor

from annotix_ml import BASE_DIR
from annotix_ml.graphtransf.data.dataset import GraphDatasetFromSMILEs
from annotix_ml.graphtransf.data.atoms_data import VALID_ELEMENTS, TYPE_EDGES


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


def test_dataset_compute_emb():
    # Open the MSG dataset
    df = pd.read_csv(BASE_DIR / "data" / "MassSpecGym.csv")

    # Select a part of the dataset
    df = df[df.fold == "train"].sample(100)

    # Create the dataset
    k = 8
    dataset = GraphDatasetFromSMILEs(df, k=k)

    # Test the elements in the dataset
    nodes, _, pos_emb = dataset[0]
    n, _ = nodes.shape

    assert pos_emb.shape == (n, k)


def test_dataset_distribution():
    # Open the MSG dataset
    df = pd.read_csv(BASE_DIR / "data" / "MassSpecGym.csv")

    # Select a part of the dataset
    df = df[df.fold == "train"].sample(100)

    # Create the dataset
    k = 8
    dataset = GraphDatasetFromSMILEs(df, k=k)

    # Check the distribution sizes
    node_distribution = dataset.node_distribution
    edge_distribution = dataset.edge_distribution

    assert node_distribution.shape == (len(VALID_ELEMENTS),)
    assert edge_distribution.shape == (len(TYPE_EDGES),)

    # Check sums are approximately 1 (use a small tolerance)
    assert torch.allclose(node_distribution.sum(), torch.tensor(1.0), atol=1e-4)
    assert torch.allclose(edge_distribution.sum(), torch.tensor(1.0), atol=1e-4)


if __name__ == "__main__":
    test_dataset_msg()
    test_dataset_compute_emb()
    test_dataset_distribution()
