from unittest.mock import patch

import torch
from rdkit import Chem

from annotix_ml.graphtransf.data.atoms_data import TYPE_EDGES, VALID_ELEMENTS
from annotix_ml.graphtransf.data.data_utils import (
    batch_graph_to_smiles,
    batch_graph_to_smiles_digress,
    graph_to_mol,
    graph_to_smiles,
    graph_to_smiles_digress,
    mask_any_tensor,
    mol_to_smiles,
)
from annotix_ml.graphtransf.test_utils import create_random_inp

ELEMENTS = list(VALID_ELEMENTS)
N_ATOMS = len(ELEMENTS)
N_BONDS = len(TYPE_EDGES)

C_IDX = ELEMENTS.index("C")
N_IDX = ELEMENTS.index("N")
O_IDX = ELEMENTS.index("O")
SINGLE_IDX = TYPE_EDGES.index(Chem.BondType.SINGLE)
DOUBLE_IDX = TYPE_EDGES.index(Chem.BondType.DOUBLE)
TRIPLE_IDX = TYPE_EDGES.index(Chem.BondType.TRIPLE)


def make_nodes(symbols):
    """One-hot node tensor (n_atoms, N_ATOMS) for a list of atom symbols."""
    nodes = torch.zeros(len(symbols), N_ATOMS)
    for i, sym in enumerate(symbols):
        nodes[i, ELEMENTS.index(sym)] = 1.0
    return nodes


def make_edges(n, bonds):
    """One-hot edge tensor (n, n, N_BONDS).

    Positions with no bond remain all-zero; argmax returns 0 = TYPE_EDGES[0] = "NoBond".
    bonds: list of (i, j, bond_type_index) tuples (symmetric, so (i,j) and (j,i) are set).
    """
    edges = torch.zeros(n, n, N_BONDS)
    for i, j, bt_idx in bonds:
        edges[i, j, bt_idx] = 1.0
        edges[j, i, bt_idx] = 1.0
    return edges


# ---------------------------------------------------------------------------
# mol_to_smiles
# ---------------------------------------------------------------------------


def test_mol_to_smiles_valid_molecule():
    """mol_to_smiles returns a canonical SMILES string for a valid RDKit mol."""
    mol = Chem.RWMol()
    mol.AddAtom(Chem.Atom("C"))
    result = mol_to_smiles(mol)
    assert result == "C"


def test_mol_to_smiles_returns_none_on_value_error():
    """mol_to_smiles returns None when SanitizeMol raises ValueError."""
    mol = Chem.RWMol()
    mol.AddAtom(Chem.Atom("C"))
    with patch.object(Chem, "SanitizeMol", side_effect=ValueError("bad mol")):
        result = mol_to_smiles(mol)
    assert result is None


def test_mol_to_smiles_returns_none_on_runtime_error():
    """mol_to_smiles returns None when SanitizeMol raises RuntimeError.

    Regression test for RDKit 2025.09.1 which raises RuntimeError
    ('Pre-condition Violation: getValence called without calcExplicitValence')
    instead of ValueError for certain invalid molecules built with RWMol.
    """
    mol = Chem.RWMol()
    mol.AddAtom(Chem.Atom("C"))
    with patch.object(
        Chem,
        "SanitizeMol",
        side_effect=RuntimeError("Pre-condition Violation"),
    ):
        result = mol_to_smiles(mol)
    assert result is None


# ---------------------------------------------------------------------------
# graph_to_mol
# ---------------------------------------------------------------------------


def test_graph_to_mol_single_atom():
    """graph_to_mol with one node creates a molecule with the correct atom symbol."""
    nodes = make_nodes(["C"])
    edges = make_edges(1, [])
    mol = graph_to_mol(nodes, edges, ELEMENTS)
    assert mol.GetNumAtoms() == 1
    assert mol.GetAtomWithIdx(0).GetSymbol() == "C"


def test_graph_to_mol_two_atoms_with_single_bond():
    """graph_to_mol produces correct atom symbols and a single bond between two atoms."""
    nodes = make_nodes(["C", "O"])
    edges = make_edges(2, [(0, 1, SINGLE_IDX)])
    mol = graph_to_mol(nodes, edges, ELEMENTS)
    assert mol.GetNumAtoms() == 2
    assert mol.GetAtomWithIdx(0).GetSymbol() == "C"
    assert mol.GetAtomWithIdx(1).GetSymbol() == "O"
    assert mol.GetNumBonds() == 1
    bond = mol.GetBondBetweenAtoms(0, 1)
    assert bond is not None
    assert bond.GetBondType() == Chem.BondType.SINGLE


def test_graph_to_mol_no_bond():
    """graph_to_mol with all-zero edge tensor produces a molecule with no bonds."""
    nodes = make_nodes(["C", "N"])
    edges = make_edges(2, [])
    mol = graph_to_mol(nodes, edges, ELEMENTS)
    assert mol.GetNumAtoms() == 2
    assert mol.GetNumBonds() == 0


def test_graph_to_mol_double_bond():
    """graph_to_mol produces a double bond when the double-bond index is set."""
    nodes = make_nodes(["C", "O"])
    edges = make_edges(2, [(0, 1, DOUBLE_IDX)])
    mol = graph_to_mol(nodes, edges, ELEMENTS)
    bond = mol.GetBondBetweenAtoms(0, 1)
    assert bond is not None
    assert bond.GetBondType() == Chem.BondType.DOUBLE


# ---------------------------------------------------------------------------
# graph_to_smiles
# ---------------------------------------------------------------------------


def test_graph_to_smiles_methane():
    """Single carbon atom converts to canonical SMILES 'C'."""
    nodes = make_nodes(["C"])
    edges = make_edges(1, [])
    smiles = graph_to_smiles(nodes, edges, ELEMENTS)
    assert smiles == "C"


def test_graph_to_smiles_ethane():
    """Two carbons with single bond convert to canonical SMILES 'CC'."""
    nodes = make_nodes(["C", "C"])
    edges = make_edges(2, [(0, 1, SINGLE_IDX)])
    smiles = graph_to_smiles(nodes, edges, ELEMENTS)
    assert smiles == "CC"


def test_graph_to_smiles_ethanol():
    """C-C-O graph converts to canonical SMILES 'CCO'."""
    nodes = make_nodes(["C", "C", "O"])
    edges = make_edges(3, [(0, 1, SINGLE_IDX), (1, 2, SINGLE_IDX)])
    smiles = graph_to_smiles(nodes, edges, ELEMENTS)
    assert smiles == "CCO"


def test_graph_to_smiles_formaldehyde():
    """C=O graph converts to canonical SMILES 'C=O'."""
    nodes = make_nodes(["C", "O"])
    edges = make_edges(2, [(0, 1, DOUBLE_IDX)])
    smiles = graph_to_smiles(nodes, edges, ELEMENTS)
    assert smiles == "C=O"


def test_graph_to_smiles_invalid_returns_none():
    """Pentavalent carbon (5 single bonds) cannot be sanitized; returns None."""
    # Central C bonded to 5 other carbons exceeds the carbon valence of 4.
    nodes = make_nodes(["C", "C", "C", "C", "C", "C"])
    bonds = [(0, i, SINGLE_IDX) for i in range(1, 6)]
    edges = make_edges(6, bonds)
    smiles = graph_to_smiles(nodes, edges, ELEMENTS)
    assert smiles is None


# ---------------------------------------------------------------------------
# graph_to_smiles_digress
# ---------------------------------------------------------------------------


def test_graph_to_smiles_digress_single_atom():
    """DiGress conversion of a single carbon gives 'C'."""
    nodes = make_nodes(["C"])
    edges = make_edges(1, [])
    smiles = graph_to_smiles_digress(nodes, edges, ELEMENTS)
    assert smiles == "C"


def test_graph_to_smiles_digress_ethane():
    """DiGress conversion of ethane gives 'CC'."""
    nodes = make_nodes(["C", "C"])
    edges = make_edges(2, [(0, 1, SINGLE_IDX)])
    smiles = graph_to_smiles_digress(nodes, edges, ELEMENTS)
    assert smiles == "CC"


def test_graph_to_smiles_digress_picks_largest_fragment():
    """DiGress keeps only the largest fragment from a disconnected graph.

    Graph: C0-C1 (single bond) with C2 disconnected.
    Expected: 'CC' (the larger 2-atom fragment, not the isolated 'C').
    """
    nodes = make_nodes(["C", "C", "C"])
    edges = make_edges(3, [(0, 1, SINGLE_IDX)])
    smiles = graph_to_smiles_digress(nodes, edges, ELEMENTS)
    assert smiles == "CC"


def test_graph_to_smiles_digress_fully_disconnected():
    """DiGress on two isolated carbons returns 'C' (one of the equal-size fragments)."""
    nodes = make_nodes(["C", "C"])
    edges = make_edges(2, [])
    smiles = graph_to_smiles_digress(nodes, edges, ELEMENTS)
    assert smiles == "C"


def test_graph_to_smiles_digress_invalid_returns_none():
    """DiGress returns None for a chemically invalid molecule (pentavalent carbon)."""
    nodes = make_nodes(["C", "C", "C", "C", "C", "C"])
    bonds = [(0, i, SINGLE_IDX) for i in range(1, 6)]
    edges = make_edges(6, bonds)
    smiles = graph_to_smiles_digress(nodes, edges, ELEMENTS)
    assert smiles is None


# ---------------------------------------------------------------------------
# batch_graph_to_smiles (existing, kept for regression)
# ---------------------------------------------------------------------------


def test_batch_graph_to_smiles():
    bs = 2
    n = 3

    nodes = torch.zeros(bs, n, N_ATOMS)
    edges = torch.zeros(bs, n, n, N_BONDS)
    mask = torch.zeros(bs, n)

    # Mol 0: C-C (ethane, 2 atoms)
    nodes[0, 0, C_IDX] = 1
    nodes[0, 1, C_IDX] = 1
    mask[0, 0] = 1
    mask[0, 1] = 1
    edges[0, 0, 1, SINGLE_IDX] = 1
    edges[0, 1, 0, SINGLE_IDX] = 1

    # Mol 1: C (methane, 1 atom)
    nodes[1, 0, C_IDX] = 1
    mask[1, 0] = 1

    smiles_list = batch_graph_to_smiles(nodes, edges, mask, ELEMENTS)
    assert len(smiles_list) == 2
    assert smiles_list[0] == "CC"
    assert smiles_list[1] == "C"


# ---------------------------------------------------------------------------
# batch_graph_to_smiles_digress
# ---------------------------------------------------------------------------


def test_batch_graph_to_smiles_digress_basic():
    """Batch DiGress converts methane and ethane to correct SMILES."""
    bs = 2
    n = 3

    nodes = torch.zeros(bs, n, N_ATOMS)
    edges = torch.zeros(bs, n, n, N_BONDS)
    mask = torch.zeros(bs, n)

    # Mol 0: methane
    nodes[0, 0, C_IDX] = 1.0
    mask[0, 0] = 1.0

    # Mol 1: ethane
    nodes[1, 0, C_IDX] = 1.0
    nodes[1, 1, C_IDX] = 1.0
    edges[1, 0, 1, SINGLE_IDX] = 1.0
    edges[1, 1, 0, SINGLE_IDX] = 1.0
    mask[1, 0] = 1.0
    mask[1, 1] = 1.0

    smiles_list = batch_graph_to_smiles_digress(nodes, edges, mask, ELEMENTS)
    assert len(smiles_list) == bs
    assert smiles_list[0] == "C"
    assert smiles_list[1] == "CC"


def test_batch_graph_to_smiles_digress_keeps_largest_fragment():
    """Batch DiGress returns only the largest fragment for a disconnected molecule."""
    bs = 1
    n = 3  # C0-C1 bonded, C2 disconnected

    nodes = torch.zeros(bs, n, N_ATOMS)
    edges = torch.zeros(bs, n, n, N_BONDS)
    mask = torch.ones(bs, n)  # all 3 atoms active

    nodes[0, 0, C_IDX] = 1.0
    nodes[0, 1, C_IDX] = 1.0
    nodes[0, 2, C_IDX] = 1.0
    edges[0, 0, 1, SINGLE_IDX] = 1.0
    edges[0, 1, 0, SINGLE_IDX] = 1.0
    # C2 has no bonds → disconnected

    smiles_list = batch_graph_to_smiles_digress(nodes, edges, mask, ELEMENTS)
    assert smiles_list[0] == "CC"


def test_batch_graph_to_smiles_digress_invalid_molecule():
    """Batch DiGress returns None for an invalid molecule (pentavalent C)."""
    bs = 1
    n = 6

    nodes = torch.zeros(bs, n, N_ATOMS)
    edges = torch.zeros(bs, n, n, N_BONDS)
    mask = torch.ones(bs, n)

    for i in range(n):
        nodes[0, i, C_IDX] = 1.0
    for i in range(1, n):
        edges[0, 0, i, SINGLE_IDX] = 1.0
        edges[0, i, 0, SINGLE_IDX] = 1.0

    smiles_list = batch_graph_to_smiles_digress(nodes, edges, mask, ELEMENTS)
    assert smiles_list[0] is None


# ---------------------------------------------------------------------------
# mask_any_tensor
# ---------------------------------------------------------------------------


def test_mask_any_tensor():
    bs = 64
    n = 54
    d = 512
    de = 256

    N, E, mask = create_random_inp(bs, n, d, de)

    N_masked = N * mask.unsqueeze(-1)
    E_masked = E * mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, n, -1)

    N_test = mask_any_tensor(N, mask)
    E_test = mask_any_tensor(E, mask)

    assert (N_masked == N_test).all()
    assert (E_masked == E_test).all()
