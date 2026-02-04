import logging
from pathlib import Path
from collections import Counter
from typing import Callable
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import rdkit.Chem as Chem
from rdkit.RDLogger import DisableLog
from pandas.core.frame import DataFrame

from annotix_ml.graphtransf.data.atoms_data import TYPE_EDGES, VALID_ELEMENTS
from annotix_ml.spectraencoder.data.featurizers import PeakFormula
from annotix_ml.spectraencoder.data.objects import Spectra


def _extract_atoms_from_smiles(smiles_list: list[str]) -> list[str]:
    atoms = set()
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            for atom in mol.GetAtoms():
                atoms.add(atom.GetSymbol())
    return sorted(list(atoms))


class GraphSpecDataset(Dataset):
    """
    GraphSpecDataset combines molecular graph transformation from SMILES
    and MS2 spectra transformation from associated files.
    """

    def __init__(
        self,
        data: str | DataFrame,
        spec_folder: str | Path,
        subform_folder: str | Path,
        smile_column: str = "smiles",
        spec_column: str = "spec",
        formula_column: str = "formula",
        instrument_column: str = "instrument",
        valid_elements: list[str] | None = None,
        sanitizer: Callable | None = None,
        **kwargs,
    ):
        super().__init__()
        if isinstance(data, (str, Path)):
            self.data = pd.read_csv(
                data, sep="\t" if str(data).endswith(".tsv") else ","
            )
        else:
            self.data = data

        self.spec_folder = Path(spec_folder)
        self.subform_folder = Path(subform_folder)
        self.smile_column = smile_column
        self.spec_column = spec_column
        self.formula_column = formula_column
        self.instrument_column = instrument_column

        # SMILES list
        smiles_list = self.data[smile_column].to_list()

        # Valid elements for graph
        if valid_elements is None:
            self.valid_elements = _extract_atoms_from_smiles(smiles_list)
        else:
            self.valid_elements = valid_elements

        # Spectra featurizer
        self.spec_featurizer = PeakFormula(subform_folder=str(subform_folder), **kwargs)

        # Graph featurizer (if needed for Batch later, but we use manual graph construction here for distributions)
        # Actually, we can reuse GraphFeaturizer._featurize or manual logic.
        # Let's use manual logic from GraphDatasetFromSMILEs to compute distributions

        self.smiles = []
        self.specs = []

        smiles_nodes_edges = []
        for i, row in tqdm(
            self.data.iterrows(), total=len(self.data), desc="Processing dataset"
        ):
            sm = row[self.smile_column]
            graph = self.smilesToGraph(sm)
            if graph is None:
                continue

            nodes, edges = graph
            # Sanitize checks
            if sanitizer:
                DisableLog("rdApp.*")
                if not sanitizer(nodes, edges, valid_elements=self.valid_elements):
                    continue

            # Check if spectra exists
            spec_name = row[self.spec_column]
            spec_file = self.spec_folder / f"{spec_name}.ms"
            if not spec_file.exists():
                logging.warning(f"Spec file {spec_file} not found, skipping.")
                continue

            smiles_nodes_edges.append(
                (
                    sm,
                    spec_name,
                    row.get(self.formula_column, ""),
                    row.get(self.instrument_column, ""),
                    nodes.sum(0).unsqueeze(0),
                    edges.sum(0).sum(0).unsqueeze(0),
                )
            )

        if not smiles_nodes_edges:
            raise ValueError("No valid SMILES/Spectra pairs found in the dataset.")

        smiles, spec_names, formulas, instruments, nodes, edges = zip(
            *smiles_nodes_edges
        )
        self.smiles = smiles
        self.spec_names = spec_names
        self.formulas = formulas
        self.instruments = instruments

        # Compute the distribution of nodes
        node_number = torch.cat(nodes).sum(0)
        self.nodes_distribution = node_number / (node_number.sum(0).item() + 1e-9)

        # Compute the distribution of edges
        edge_number = torch.cat(edges).sum(0)
        self.edges_distribution = edge_number / (edge_number.sum(0).item() + 1e-9)

        # Compute the distribution of number of nodes
        num_atoms_dist = Counter([n.sum(-1).item() for n in nodes])
        max_num_atom = max(num_atoms_dist.keys()) if num_atoms_dist else 0
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
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        n = mol.GetNumAtoms()
        nodes = torch.zeros((n, len(self.valid_elements)), dtype=int)
        edges = torch.zeros((n, n, len(TYPE_EDGES)), dtype=int)

        for i, atom in enumerate(mol.GetAtoms()):
            atom_symbol = atom.GetSymbol()
            if atom_symbol not in self.valid_elements:
                return None
            nodes[i, self.valid_elements.index(atom_symbol)] = 1

            for bond in atom.GetBonds():
                s = bond.GetBeginAtomIdx()
                e = bond.GetEndAtomIdx()
                bond_type = bond.GetBondType()
                if bond_type not in TYPE_EDGES:
                    return None
                edges[s, e, TYPE_EDGES.index(bond_type)] = 1

        edges = edges + edges.transpose(0, 1)
        mask = edges.sum(-1)
        edges[:, :, 0] += 1 - mask
        return nodes, edges

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        sm = self.smiles[idx]
        spec_name = self.spec_names[idx]
        formula = self.formulas[idx]
        instrument = self.instruments[idx]

        # Graph
        nodes, edges = self.smilesToGraph(sm)

        # Spectra
        spec_obj = Spectra(
            spectra_name=spec_name,
            spectra_file=str(self.spec_folder / f"{spec_name}.ms"),
            spectra_formula=formula,
            instrument=instrument,
        )
        spec_features = self.spec_featurizer.featurize(spec_obj)

        # Return merged dict
        res = {"nodes": nodes, "edges": edges, "smiles": sm, "spec_name": spec_name}
        res.update(spec_features)
        return res
