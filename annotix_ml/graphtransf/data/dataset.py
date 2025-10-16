import rdkit.Chem as Chem
import torch
from torch.utils.data import Dataset

import pandas as pd
from pandas.core.frame import DataFrame

from annotix_ml.graphtransf.data.atoms_data import VALID_ELEMENTS, TYPE_EDGES
from annotix_ml.graphtransf.math.positional_emb import laplacian_embedding


class GraphDatasetFromSMILEs(Dataset):
    def __init__(
        self,
        data: str | DataFrame,
        smile_column: str = "smiles",
        k: int | None = None,
    ):
        super().__init__()
        if isinstance(data, str):
            data = pd.read_csv(data)

        elif isinstance(data, DataFrame):
            pass

        else:
            raise TypeError(
                f"{type(data)} is not accepted to instantiate GraphDataset."
            )
        
        # Register the number of eigenvectors to take
        self.k = k

        # Get the valid smiles, and compute node / edges dist -> for noise schedule
        smiles_nodes_edges = [
            (sm, graph[0].sum(0).unsqueeze(0), graph[1].sum(0).sum(0).unsqueeze(0)) # graph[0]=nodes, graph[1]=edges
            for sm in data[smile_column].to_list()
            if (graph := self.smilesToGraph(sm))
        ]
        smiles, nodes, edges = zip(*smiles_nodes_edges)
        self.smiles = smiles

        # Compute the distribution of nodes
        node_number = torch.cat(nodes).sum(0) # (natoms)
        node_distribution = node_number / (node_number.sum(0).item())
        self.node_distribution = node_distribution

        # Compute the distribution of the edges
        edge_number = torch.cat(edges).sum(0) # (nbonds)
        edge_distribution = edge_number / (edge_number.sum(0).item())
        self.edge_distribution = edge_distribution

    @staticmethod
    def smilesToGraph(smiles: str):
        # Make a molecule
        mol = Chem.MolFromSmiles(smiles)
        
        # If rdkit is not able to make a molecule from the smile -> None
        if mol is None:
            return None
        
        # Initialize the nodes, edges matrices
        n = mol.GetNumAtoms()
        nodes = torch.zeros((n, len(VALID_ELEMENTS)), dtype=int) # (n, natoms)
        edges = torch.zeros((n, n, len(TYPE_EDGES)), dtype=int) # (n, n, nbonds)
        
        for i, atom in enumerate(mol.GetAtoms()):
            # Return None for the molecules that are not in the ones of interest
            atom_symbol = atom.GetSymbol()
            if atom_symbol not in VALID_ELEMENTS:
                return None
            
            # One-hot encode the atom
            nodes[i, VALID_ELEMENTS.index(atom_symbol)] = 1

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
        edges = edges + edges.transpose(0, 1) # (n, natoms)
        
        # One-hot encode the edges that have yet no 1 -> means no edges
        mask = edges.sum(-1)
        edges[:, :, 0] += 1 - mask # Put a 0 at dim1 -> No Bond # (n, n, nbonds)

        return nodes, edges
    
    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        sm = self.smiles[idx]
        
        # Calculate the nodes and edges
        nodes, edges = self.smilesToGraph(sm) # (n, natoms), (n, n, nbonds)
        
        # Compute the laplacian
        if self.k:
            if isinstance(self.k, int):
                pos_emb = laplacian_embedding(edges, self.k) # (n, k)
            
            else:
                raise TypeError(
                    f"Gave k value but with wrong type: {type(self.k)}"
                )

            return nodes, edges, pos_emb
        
        else:
            return nodes, edges
