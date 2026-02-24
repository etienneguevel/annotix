import torch
import rdkit.Chem as Chem
from annotix_ml.data.atoms_data import TYPE_EDGES


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
    while mask.dim() < t.dim():
        mask = mask.unsqueeze(-1)

    return t.masked_fill(~mask.bool(), fill)


def mol_to_smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except (ValueError, RuntimeError):
        return None
    return Chem.MolToSmiles(mol)


def graph_to_mol(nodes, edges, valid_elements):
    n_atoms = nodes.shape[0]
    # Create a writable molecule
    mol = Chem.RWMol()

    # Add atoms
    atom_indices = []
    for j in range(n_atoms):
        atom_idx = torch.argmax(nodes[j]).item()
        atom_symbol = valid_elements[atom_idx]
        atom = Chem.Atom(atom_symbol)
        idx = mol.AddAtom(atom)
        atom_indices.append(idx)

    # Add bonds
    # Iterate over the upper triangle to avoid duplicates

    for j in range(n_atoms):
        for k in range(j + 1, n_atoms):
            bond_type_idx = torch.argmax(edges[j, k]).item()
            bond_type = TYPE_EDGES[bond_type_idx]

            if bond_type != "NoBond":
                mol.AddBond(atom_indices[j], atom_indices[k], bond_type)

                if bond_type == Chem.BondType.AROMATIC:
                    mol.GetAtomWithIdx(atom_indices[j]).SetIsAromatic(True)
                    mol.GetAtomWithIdx(atom_indices[k]).SetIsAromatic(True)

    return mol


def auto_fix_kekulization(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)

    # 1) initialize valence / implicit Hs (but do NOT kekulize)
    Chem.SanitizeMol(
        mol,
        sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
        ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
        ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY,
    )

    # 2) now it's safe to (re)compute aromaticity
    Chem.SetAromaticity(mol)

    # 3) heuristic: assign one pyrrolic N
    aromatic_ns = [
        a
        for a in mol.GetAtoms()
        if a.GetSymbol() == "N" and a.GetIsAromatic() and a.GetDegree() == 2
    ]

    # Try adding H to each candidate N
    for n in aromatic_ns:
        n.SetNumExplicitHs(1)
        n.SetNoImplicit(True)

        # 4) full sanitize (now kekulization works?)
        try:
            Chem.SanitizeMol(mol)
            return mol
        except Chem.rdchem.KekulizeException:
            # Revert and try next
            n.SetNumExplicitHs(0)
            n.SetNoImplicit(False)

    # If loop finishes without success, return None
    return None


def graph_to_smiles(
    nodes: torch.Tensor,
    edges: torch.Tensor,
    valid_elements: list[str],
) -> str | None:
    """
    Convert a single graph into a SMILES string.

    Args:
    - nodes: torch.Tensor, one-hot encoded nodes (natoms)
    - edges: torch.Tensor, one-hot encoded edges (natoms, natoms, nbonds)
    - valid_elements: list[str], list of valid element symbols

    Returns:
    - smiles: str | None, reconstructed SMILES string. None if invalid.
    """
    # Create a writable molecule
    mol = graph_to_mol(nodes, edges, valid_elements)

    # Sanitize the molecule
    smiles = None
    try:
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol)
    except Exception:
        try:
            Chem.SanitizeMol(
                mol,
                Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
            )
            smiles = Chem.MolToSmiles(mol)
        except Exception:
            pass

    if smiles is None:
        return None

    mol = Chem.MolFromSmiles(smiles)

    if mol:
        smiles = Chem.MolToSmiles(mol)
    else:
        try:
            mol = auto_fix_kekulization(smiles)
            smiles = Chem.MolToSmiles(mol)
        except:
            smiles = None

    return smiles


def batch_graph_to_smiles(
    nodes: torch.Tensor,
    edges: torch.Tensor,
    mask: torch.Tensor,
    valid_elements: list[str],
) -> list[str | None]:
    """
    Convert a batch of graphs into a list of SMILES strings.

    Args:
    - nodes: torch.Tensor, one-hot encoded nodes (bs, n, natoms)
    - edges: torch.Tensor, one-hot encoded edges (bs, n, n, nbonds)
    - mask: torch.Tensor, binary mask indicating valid nodes (bs, n)
    - valid_elements: list[str], list of valid element symbols

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

        # Convert to SMILES using the helper function
        smiles = graph_to_smiles(current_nodes, current_edges, valid_elements)

        smiles_list.append(smiles)

    return smiles_list


def graph_to_smiles_digress(
    nodes: torch.Tensor,
    edges: torch.Tensor,
    valid_elements: list[str],
) -> str | None:
    """
    Convert a single graph to SMILES using Digress method (checks for fragments).
    """
    # Create a writable molecule
    mol = graph_to_mol(nodes, edges, valid_elements)
    smiles = mol_to_smiles(mol)

    try:
        mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    except Exception:
        pass

    if smiles is not None:
        try:
            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
            smiles = mol_to_smiles(largest_mol)
            return smiles

        except Chem.rdchem.AtomValenceException:
            print("Valence error in GetmolFrags")
            return None

        except Chem.rdchem.KekulizeException:
            print("Can't kekulize molecule")
            return None

    return None


def batch_graph_to_smiles_digress(
    nodes: torch.Tensor,
    edges: torch.Tensor,
    mask: torch.Tensor,
    valid_elements: list[str],
) -> list[str | None]:
    """
    Convert a batch of graphs into a list of SMILES strings using Digress method.
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

        # Convert using helper
        smiles = graph_to_smiles_digress(current_nodes, current_edges, valid_elements)
        smiles_list.append(smiles)

    return smiles_list
