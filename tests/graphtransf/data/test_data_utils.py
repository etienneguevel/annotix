import torch
from rdkit import Chem

from annotix_ml.graphtransf.data.atoms_data import VALID_ELEMENTS, TYPE_EDGES
from annotix_ml.graphtransf.data.data_utils import (
    mask_any_tensor,
    batch_graph_to_smiles,
)
from annotix_ml.graphtransf.test_utils import create_random_inp


def test_batch_graph_to_smiles():
    # Create a simple mock batch
    # Molecule 1: C-C (Ethane-like but simplified)
    # Molecule 2: C (Methane-like)

    # Nodes: (bs, n, natoms)
    bs = 2
    n = 3  # Max atoms
    natoms = len(list(VALID_ELEMENTS))
    nbonds = len(TYPE_EDGES)

    nodes = torch.zeros((bs, n, natoms))
    edges = torch.zeros((bs, n, n, nbonds))
    mask = torch.zeros((bs, n))

    # Mol 1: 2 atoms
    c_idx = list(VALID_ELEMENTS).index("C")
    nodes[0, 0, c_idx] = 1
    nodes[0, 1, c_idx] = 1
    mask[0, 0] = 1
    mask[0, 1] = 1

    # Bond between 0 and 1 (Single)
    single_bond_idx = TYPE_EDGES.index(Chem.BondType.SINGLE)
    edges[0, 0, 1, single_bond_idx] = 1
    edges[0, 1, 0, single_bond_idx] = 1  # Symmetric

    # Mol 2: 1 atom
    nodes[1, 0, c_idx] = 1
    mask[1, 0] = 1

    # Run function
    smiles_list = batch_graph_to_smiles(nodes, edges, mask, list(VALID_ELEMENTS))
    assert len(smiles_list) == 2

    # Expected SMILES might vary slightly depending on canonicalization, but C-C is CC and C is C
    assert smiles_list[0] == "CC"
    assert smiles_list[1] == "C"
    print("Test passed!")


def test_mask_any_tensor():
    # Choose some dimensions
    bs = 64
    n = 54
    d = 512
    de = 256

    # Create random inputs
    N, E, mask = create_random_inp(bs, n, d, de)

    # Do the masking with size matchings
    N_masked = N * mask.unsqueeze(-1)
    E_masked = E * mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, n, -1)

    # obtain the masked inputs with mask_any_tensor
    N_test = mask_any_tensor(N, mask)
    E_test = mask_any_tensor(E, mask)

    assert (N_masked == N_test).all()
    assert (E_masked == E_test).all()
