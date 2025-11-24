import torch
import rdkit.Chem as Chem
from annotix_ml.graphtransf.data.atoms_data import VALID_ELEMENTS, TYPE_EDGES


def mask_any_tensor(
    t: torch.Tensor, mask: torch.Tensor, fill: float = 0.0
) -> torch.Tensor:
    """Apply a 0/1 mask to the leading dimensions of a tensor, filling masked
    positions with a scalar value.

    The function expects ``mask`` to have shape equal to the leading (left-most)
    dimensions of ``t``. Any remaining trailing dimensions of ``t`` are treated
    as feature dimensions and the mask is expanded (unsqueezed) over them.

    Behavior summary
    - If ``mask.shape == t.shape[:len(mask.shape)]`` the mask is unsqueezed
      for each remaining trailing dimension of ``t`` and then applied.
    - Positions where ``mask == 0`` are replaced by ``fill`` via
      :meth:`torch.Tensor.masked_fill`.
    - Non-zero mask values are treated as keep indicators.

    Args:
        t: Input tensor to mask. Any dtype is supported; ``fill`` will be
           cast as needed by PyTorch when filling.
        mask: Binary (0/1) or boolean tensor whose shape must match the leading
           dimensions of ``t``. For example, if ``t`` has shape
           ``(B, N, F)`` then ``mask`` can be ``(B, N)`` or ``(B, N, 1)``.
        fill: Scalar value used to fill masked locations (where ``mask == 0``).

    Returns:
        A tensor of the same shape and dtype as ``t`` with masked positions
        replaced by ``fill``.

    Example:
        >>> t = torch.tensor([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])
        >>> mask = torch.tensor([[1, 0], [0, 1]])
        >>> mask_any_tensor(t, mask, fill=0.0)
        tensor([[[1., 2.],
                 [0., 0.]],

                [[0., 0.],
                 [7., 8.]]])

    Notes and edge cases:
    - The function asserts that ``mask.shape`` equals the leading dimensions of
      ``t``; a mismatch will raise an AssertionError.
    - ``mask`` may be integer or boolean. Zeros are considered masked.
    - Time/space complexity is linear in the number of elements in ``t``.

    """
    # Assert that the dimensions match
    mask_dim = mask.shape
    tensor_dim = t.shape
    assert mask_dim == tensor_dim[: len(mask_dim)], (
        "mask dimension doesn't match tensor",
        mask_dim,
        tensor_dim,
    )

    # Expand the mask tensor
    expand_dims = tensor_dim[len(mask_dim) :]
    for _ in expand_dims:
        mask = mask.unsqueeze(-1)

    # Fill where the mask is equal to 0
    t = t.masked_fill(mask == 0, fill)

    return t


def batch_graph_to_smiles(
    nodes: torch.Tensor, edges: torch.Tensor, mask: torch.Tensor
) -> list[str | None]:
    """
    Convert a batch of graphs into a list of SMILES strings.

    Args:
    - nodes: torch.Tensor, one-hot encoded nodes (bs, n, natoms)
    - edges: torch.Tensor, one-hot encoded edges (bs, n, n, nbonds)
    - mask: torch.Tensor, binary mask indicating valid nodes (bs, n)

    Returns:
    - smiles_list: list[str | None], list of reconstructed SMILES strings. None if invalid.
    """
    smiles_list = []
    bs = nodes.shape[0]

    for i in range(bs):
        # Determine the number of atoms for this graph
        n_atoms = int(mask[i].sum().item())

        # Slice the nodes and edges
        # nodes: (n, natoms) -> (n_atoms, natoms)
        current_nodes = nodes[i, :n_atoms]

        # edges: (n, n, nbonds) -> (n_atoms, n_atoms, nbonds)
        current_edges = edges[i, :n_atoms, :n_atoms]

        # Create a writable molecule
        mol = Chem.RWMol()

        # Add atoms
        atom_indices = []
        for j in range(n_atoms):
            atom_idx = torch.argmax(current_nodes[j]).item()
            atom_symbol = VALID_ELEMENTS[atom_idx]
            atom = Chem.Atom(atom_symbol)
            idx = mol.AddAtom(atom)
            atom_indices.append(idx)

        # Add bonds
        # Iterate over the upper triangle to avoid duplicates
        for j in range(n_atoms):
            for k in range(j + 1, n_atoms):
                bond_type_idx = torch.argmax(current_edges[j, k]).item()
                bond_type = TYPE_EDGES[bond_type_idx]

                if bond_type != "NoBond":
                    mol.AddBond(atom_indices[j], atom_indices[k], bond_type)

        # Sanitize the molecule
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            smiles_list.append(None)
            continue

        # Convert to SMILES
        smiles = Chem.MolToSmiles(mol)
        smiles_list.append(smiles)

    return smiles_list
