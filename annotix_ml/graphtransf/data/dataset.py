from pathlib import PosixPath

import pandas as pd
import rdkit.Chem as Chem
import torch
from pandas.core.frame import DataFrame
from torch.utils.data import Dataset

from annotix_ml.graphtransf.data.atoms_data import TYPE_EDGES


def _extract_atoms_from_smiles(smiles_list: list[str]) -> list[str]:
    """
    Extract all unique atom symbols from a list of SMILES strings.

    Args:
    - smiles_list: list[str], list of SMILES strings to analyze

    Returns:
    - list[str], sorted list of unique atom symbols found in the SMILES
    """
    atoms = set()
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            for atom in mol.GetAtoms():
                atoms.add(atom.GetSymbol())
    return sorted(list(atoms))


class GraphDatasetFromSMILEs(Dataset):
    """
    Build a torch Dataset from a csv files having one column indicating the
    SMILEs of molecules. The molecules are filtered to only include the ones
    with the atoms within the VALID_ELEMENTS constant or a custom list.
    The nodes and edges distributions of the data are also computed to later
    be used for the noise model.
    """

    def __init__(
        self,
        data: str | DataFrame,
        smile_column: str = "smiles",
        split: str | None = None,
        split_column: str | None = None,
        valid_elements: list[str] | None = None,
    ):
        """
        Args:
        - data: str | DataFrame, either the path to the csv or the csv opened as
        a pd.DataFrame.
        - smile_column: str = "smiles", the column in which the smiles are
        contained.
        - split: str | None = None, the name of the split to select to build the
        dataset.
        - split_column: str | None = None, the name of the column where to search
        the split of the row.
        - valid_elements: list[str] | None = None, list of valid atom symbols to use.
        If None, automatically extracts atoms from the SMILES in the dataset.

        Returns:
        This describes here the __getitem__ method of this object. At index idx
        we get the one-hot encoded nodes and edges vectors resp. of sizes (n, natoms) and
        (n, n, nbonds) where n is the number of heavy atoms within the SMILEs,
        natoms is the length of VALID_ATOMS and nbonds is the length of TYPE_EDGES.
        """
        super().__init__()
        if isinstance(data, (str, PosixPath)):
            data = pd.read_csv(data)

        elif isinstance(data, DataFrame):
            pass

        else:
            raise TypeError(
                f"{type(data)} is not accepted to instantiate GraphDataset."
            )

        if split:
            data = data[data[split_column] == split]

        # Extract SMILES list
        smiles_list = data[smile_column].to_list()

        # Determine valid elements to use
        if valid_elements is None:
            # Auto-detect atoms from the dataset
            self.valid_elements = _extract_atoms_from_smiles(smiles_list)
        else:
            self.valid_elements = valid_elements

        # Get the valid smiles, and compute node / edges distributions -> for noise schedule
        smiles_nodes_edges = [
            (
                sm,
                graph[0].sum(0).unsqueeze(0),
                graph[1].sum(0).sum(0).unsqueeze(0),
            )  # graph[0]=nodes, graph[1]=edges
            for sm in smiles_list
            if (graph := self.smilesToGraph(sm))
        ]
        smiles, nodes, edges = zip(*smiles_nodes_edges)
        self.smiles = smiles

        # Compute the distribution of nodes
        node_number = torch.cat(nodes).sum(0)  # (natoms)
        node_distribution = node_number / (node_number.sum(0).item())
        self.nodes_distribution = node_distribution

        # Compute the distribution of the edges
        edge_number = torch.cat(edges).sum(0)  # (nbonds)
        edge_distribution = edge_number / (edge_number.sum(0).item())
        self.edges_distribution = edge_distribution

    def smilesToGraph(self, smiles: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert a smiles into its node and edges representation as tensors.

        Args:
        - smiles: str, the smiles string to convert

        Returns:
        The nodes and edges tensor resp. of sizes (n, natoms) and (n, n, nbonds).
        """
        # Make a molecule
        mol = Chem.MolFromSmiles(smiles)

        # If rdkit is not able to make a molecule from the smile -> None
        if mol is None:
            return None

        # Initialize the nodes, edges matrices
        n = mol.GetNumAtoms()
        nodes = torch.zeros((n, len(self.valid_elements)), dtype=int)  # (n, natoms)
        edges = torch.zeros((n, n, len(TYPE_EDGES)), dtype=int)  # (n, n, nbonds)

        for i, atom in enumerate(mol.GetAtoms()):
            # Return None for the molecules that are not in the ones of interest
            atom_symbol = atom.GetSymbol()
            if atom_symbol not in self.valid_elements:
                return None

            # One-hot encode the atom
            nodes[i, self.valid_elements.index(atom_symbol)] = 1

            # Iter over the bonds of the atom
            for bond in atom.GetBonds():
                # TODO : check how this method works -> doubt that bond works sym
                s = bond.GetBeginAtomIdx()
                e = bond.GetEndAtomIdx()
                bond_type = bond.GetBondType()

                # Return None for the molecules with edges not in the ones of interest.
                if bond_type not in TYPE_EDGES:
                    return None

                # One-hot encode the type of edge
                edges[s, e, TYPE_EDGES.index(bond_type)] = 1

        # Symmetrize the matrix
        edges = edges + edges.transpose(0, 1)  # (n, natoms)

        # One-hot encode the edges that have yet no 1 -> means no edges
        mask = edges.sum(-1)
        edges[:, :, 0] += 1 - mask  # Put a 0 at dim1 -> No Bond # (n, n, nbonds)

        return nodes, edges

    def graphToSmiles(self, nodes: torch.Tensor, edges: torch.Tensor) -> str:
        """
        Convert a graph representation (nodes and edges) back to a SMILES string.

        Args:
        - nodes: torch.Tensor, one-hot encoded nodes (n, natoms)
        - edges: torch.Tensor, one-hot encoded edges (n, n, nbonds)

        Returns:
        - smiles: str, the reconstructed SMILES string
        """
        # Create a writable molecule
        mol = Chem.RWMol()

        # Add atoms
        atom_indices = []
        for i in range(nodes.shape[0]):
            atom_idx = torch.argmax(nodes[i]).item()
            atom_symbol = self.valid_elements[atom_idx]
            atom = Chem.Atom(atom_symbol)
            idx = mol.AddAtom(atom)
            atom_indices.append(idx)

        # Add bonds
        # edges is (n, n, nbonds)
        # We iterate over the upper triangle to avoid duplicates
        n = nodes.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                bond_type_idx = torch.argmax(edges[i, j]).item()
                bond_type = TYPE_EDGES[bond_type_idx]

                if bond_type != "NoBond":
                    mol.AddBond(atom_indices[i], atom_indices[j], bond_type)

        # Sanitize the molecule to handle aromaticity etc.
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            # If sanitization fails, we might return a raw SMILES or None
            # For now let's try to return what we have, but it might be invalid
            pass

        # Convert to SMILES
        smiles = Chem.MolToSmiles(mol)
        return smiles

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        """
        Args:
        - idx: int, the index of the dataset to fetch

        Returns:
        One-hot encoded nodes and edges vectors resp. of sizes (n, natoms) and
        (n, n, nbonds) where n is the number of heavy atoms within the SMILEs,
        natoms is the length of VALID_ATOMS and nbonds is the length of TYPE_EDGES.
        """
        sm = self.smiles[idx]

        # Calculate the nodes and edges
        nodes, edges = self.smilesToGraph(sm)  # (n, natoms), (n, n, nbonds)

        return nodes, edges
