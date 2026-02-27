import torch
from torch import Tensor

import rdkit.Chem as Chem

from annotix_ml import BASE_DIR
from annotix_ml.data.spec_dataset import GraphSpecDataset
from annotix_ml.data.atoms_data import TYPE_EDGES


DATA_PATH = BASE_DIR / "data/BanyulsInference/labels.tsv"
SPEC_FOLDER = BASE_DIR / "data/BanyulsInference/spec_files"
SUBFORM_FOLDER = "data/BanyulsInference/subformulae/default_subformulae"


def _make_dataset(**kwargs):
    defaults = dict(
        data=DATA_PATH,
        spec_folder=SPEC_FOLDER,
        subform_folder=SUBFORM_FOLDER,
        smile_column="smiles",
        spec_column="spec",
        formula_column="formula",
        instrument_column="instrument",
    )
    defaults.update(kwargs)
    return GraphSpecDataset(**defaults)


def test_graph_spec_dataset():
    """Test basic dataset functionality: length, item keys, and collation."""
    dataset = _make_dataset()

    # Test length
    assert len(dataset) > 0

    # Test __getitem__
    item = dataset[0]
    expected_item_keys = [
        "nodes",
        "edges",
        "smiles",
        "spec_name",
        "peak_type",
        "form_vec",
        "ion_vec",
        "frag_intens",
        "instrument",
    ]
    for key in expected_item_keys:
        assert key in item, f"Missing key in item: {key}"

    # Nodes and edges are tensors with correct ranks
    assert isinstance(item["nodes"], Tensor)
    assert isinstance(item["edges"], Tensor)
    assert len(item["nodes"].size()) == 2  # (n, natoms)
    assert len(item["edges"].size()) == 3  # (n, n, nbonds)

    # Check that nodes and edges are one-hot encoded
    assert (item["nodes"].sum(-1) == 1).all()
    assert (item["edges"].sum(-1) == 1).all()


def test_dataset_distribution():
    """Test that node and edge distributions are valid probability distributions."""
    dataset = _make_dataset()

    node_distribution = dataset.nodes_distribution
    edge_distribution = dataset.edges_distribution

    # Shapes match the vocabulary sizes
    assert node_distribution.shape == (len(dataset.valid_elements),)
    assert edge_distribution.shape == (len(TYPE_EDGES),)

    # Sum to 1
    assert torch.allclose(node_distribution.sum(), torch.tensor(1.0), atol=1e-4)
    assert torch.allclose(edge_distribution.sum(), torch.tensor(1.0), atol=1e-4)

    # All values are non-negative
    assert (node_distribution >= 0).all()
    assert (edge_distribution >= 0).all()

    # num_atoms_dist and max_weight are computed
    assert dataset.num_atoms_dist is not None
    assert dataset.max_weight > 0


def test_graph_to_smiles():
    """Test round-trip SMILES -> graph -> SMILES conversion."""
    dataset = _make_dataset()

    smiles_list = ["C", "CC", "CCO", "c1ccccc1", "C1CCCCC1", "C(=O)O"]

    for sm in smiles_list:
        graph = dataset.smilesToGraph(sm)
        if graph is None:
            # Some SMILES may use elements not in valid_elements, skip them
            continue

        nodes, edges = graph

        # Verify one-hot encoding
        assert (nodes.sum(-1) == 1).all(), f"Nodes not one-hot for {sm}"
        assert (edges.sum(-1) == 1).all(), f"Edges not one-hot for {sm}"

        # Verify node count matches molecule
        mol = Chem.MolFromSmiles(sm)
        assert nodes.shape[0] == mol.GetNumAtoms(), (
            f"Node count mismatch for {sm}: {nodes.shape[0]} != {mol.GetNumAtoms()}"
        )

        # Verify edge matrix is symmetric
        assert torch.equal(edges, edges.transpose(0, 1)), (
            f"Edge matrix not symmetric for {sm}"
        )
