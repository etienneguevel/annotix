import os
from collections import Counter
from typing import Callable
from pathlib import PosixPath

import pandas as pd
import rdkit.Chem as Chem
from rdkit.RDLogger import DisableLog  # pyright: ignore[reportAttributeAccessIssue]
import torch
from pandas.core.frame import DataFrame
from torch.utils.data import Dataset
from tqdm import tqdm

from annotix_ml.graphtransf.data.atoms_data import TYPE_EDGES, VALID_ELEMENTS


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
    Torch Dataset for molecular graphs built from SMILES strings.

    The dataset filters molecules to include only those with atoms present in a
    predefined list of valid elements. It also computes the distribution of nodes
    and edges in the data, which can be used for noise scheduling in diffusion models.
    """

    def __init__(
        self,
        data: str | DataFrame,
        smile_column: str = "smiles",
        split: str | None = None,
        split_column: str | None = None,
        valid_elements: list[str] | None = None,
        sanitizer: Callable | None = None,
        verbose: bool = True,
        cache_path: str | None = None,
        save_cache: bool = True,
    ):
        """
        Initialize the GraphDatasetFromSMILEs.

        Args:
            data (str | DataFrame): Path to a CSV file or a pandas DataFrame containing the data.
            smile_column (str): Name of the column containing SMILES strings. Defaults to "smiles".
            split (str, optional): Name of the split to filter by (e.g., 'train', 'test').
            split_column (str, optional): Name of the column used for split filtering.
            valid_elements (list[str], optional): List of valid atom symbols. If None, elements
                are automatically extracted from the dataset.
            sanitizer (Callable, optional): A function to further filter molecules based on
                their graph representation.
            cache_path (str, optional): Path to a .pt cache file. If the file exists it is
                loaded directly; otherwise the dataset is built and saved there.
            save_cache (bool): Whether to write the cache file after building. Set to False
                on non-main ranks in distributed training to avoid concurrent writes.
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

        # Try to load from cache if a cache path is provided
        if cache_path is not None and os.path.isfile(cache_path):
            if verbose:
                print(f"Loading dataset from cache: {cache_path}")
            cache = torch.load(cache_path, weights_only=False)
            self.smiles = cache["smiles"]
            self.valid_elements = cache["valid_elements"]
            self.nodes_distribution = cache["nodes_distribution"]
            self.edges_distribution = cache["edges_distribution"]
            self.num_atoms_dist = cache["num_atoms_dist"]
            self.max_weight = cache["max_weight"]
            return

        self._build(smiles_list, valid_elements, sanitizer, verbose)

        # Save to cache (only if requested, e.g. main rank in distributed mode)
        if cache_path is not None and save_cache:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            if verbose:
                print(f"Saving dataset cache to: {cache_path}")
            torch.save(
                {
                    "smiles": self.smiles,
                    "valid_elements": self.valid_elements,
                    "nodes_distribution": self.nodes_distribution,
                    "edges_distribution": self.edges_distribution,
                    "num_atoms_dist": self.num_atoms_dist,
                    "max_weight": self.max_weight,
                },
                cache_path,
            )

    def _build(
        self,
        smiles_list: list[str],
        valid_elements: list[str] | None,
        sanitizer: Callable | None,
        verbose: bool,
    ) -> None:
        """Build graph representations and statistics from a list of SMILES strings."""
        # Determine valid elements to use
        if valid_elements is None:
            # Auto-detect atoms from the dataset
            self.valid_elements = _extract_atoms_from_smiles(smiles_list)
        else:
            self.valid_elements = valid_elements

        # Get the valid smiles, and compute node / edges distributions -> for noise schedule
        smiles_nodes_edges = []
        for sm in tqdm(smiles_list, desc="Building graph", disable=not verbose):
            graph = self.smilesToGraph(sm)
            if graph is None:
                continue

            nodes, edges = graph
            # Sanitize checks
            if sanitizer:
                DisableLog("rdApp.*")
                if not sanitizer(nodes, edges, valid_elements=self.valid_elements):
                    continue

            # Compute aggregated stats for distribution calculation later
            # graph[0]=nodes, graph[1]=edges
            smiles_nodes_edges.append(
                (
                    sm,
                    nodes.sum(0).unsqueeze(0),
                    edges.sum(0).sum(0).unsqueeze(0),
                )
            )

        if sanitizer:
            print(
                f"Sanitizer removed {len(smiles_list) - len(smiles_nodes_edges)} molecules / {len(smiles_list)} molecules"
            )

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

        # Compute the distribution of number of nodes
        num_atoms_dist = Counter([n.sum(-1).item() for n in nodes])
        max_num_atom = max(num_atoms_dist.keys())
        self.num_atoms_dist = torch.tensor(
            [num_atoms_dist.get(i + 1, 0) / len(nodes) for i in range(max_num_atom)]
        )

        # Compute the maximum weight of the dataset
        weight_tensor = torch.tensor(
            [getattr(VALID_ELEMENTS[at], "weight") for at in self.valid_elements]
        )
        self.max_weight = max(
            [(n_mat.float() @ weight_tensor.float()).sum().item() for n_mat in nodes]
        )

    def smilesToGraph(self, smiles: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        """
        Convert a SMILES string into node and edge tensor representations.

        Args:
            smiles (str): The SMILES string to convert.

        Returns:
            tuple[torch.Tensor, torch.Tensor] | None: A tuple (nodes, edges) if successful,
                where nodes is of shape (n, natoms) and edges is of shape (n, n, nbonds).
                Returns None if conversion fails or if the molecule contains invalid elements.
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
        Convert a graph representation back to a SMILES string.

        Args:
            nodes (torch.Tensor): One-hot encoded node features of shape (n, natoms).
            edges (torch.Tensor): One-hot encoded edge features of shape (n, n, nbonds).

        Returns:
            str: The reconstructed SMILES string.
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
        """
        Get the number of molecules in the dataset.

        Returns:
            int: The number of SMILES strings in the dataset.
        """
        return len(self.smiles)

    def __getitem__(self, idx):
        """
        Get a graph representation of a molecule by index.

        Args:
            idx (int): The index of the molecule to fetch.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple (nodes, edges) containing:
                - nodes (torch.Tensor): One-hot encoded nodes of shape (n, natoms).
                - edges (torch.Tensor): One-hot encoded edges of shape (n, n, nbonds).
        """
        sm = self.smiles[idx]

        # Calculate the nodes and edges
        nodes, edges = self.smilesToGraph(sm)  # (n, natoms), (n, n, nbonds)

        return nodes, edges
