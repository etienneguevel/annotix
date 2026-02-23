import logging
import os
from pathlib import Path
from collections import Counter
from typing import Callable

import pandas as pd
import rdkit.Chem as Chem
import torch
from pandas.core.frame import DataFrame
from rdkit.RDLogger import DisableLog
from torch.utils.data import Dataset
from tqdm import tqdm

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
        verbose: bool = True,
        cache_path: str | None = None,
        save_cache: bool = True,
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

        # Try to load from cache if a cache path is provided
        if cache_path is not None and os.path.isfile(cache_path):
            print(f"Loading dataset from cache: {cache_path}")
            cache = torch.load(cache_path, weights_only=False)
            self.smiles = cache["smiles"]
            self.spec_names = cache["spec_names"]
            self.formulas = cache["formulas"]
            self.instruments = cache["instruments"]
            self.valid_elements = cache["valid_elements"]
            self.nodes_distribution = cache["nodes_distribution"]
            self.edges_distribution = cache["edges_distribution"]
            self.num_atoms_dist = cache["num_atoms_dist"]
            self.max_weight = cache["max_weight"]

        else:
            smiles, spec_names, formulas, instruments, nodes, edges = self._build(
                sanitizer, verbose
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
                [
                    (n_mat.float() @ weight_tensor.float()).sum().item()
                    for n_mat in nodes
                ]
            )

            # Save to cache (only if requested, e.g. main rank in distributed mode)
            if cache_path is not None and save_cache:
                os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
                print(f"Saving dataset cache to: {cache_path}")
                torch.save(
                    {
                        "smiles": self.smiles,
                        "spec_names": self.spec_names,
                        "formulas": self.formulas,
                        "instruments": self.instruments,
                        "valid_elements": self.valid_elements,
                        "nodes_distribution": self.nodes_distribution,
                        "edges_distribution": self.edges_distribution,
                        "num_atoms_dist": self.num_atoms_dist,
                        "max_weight": self.max_weight,
                    },
                    cache_path,
                )

    def _build(self, sanitizer: Callable | None, verbose: bool):
        smiles_nodes_edges = []
        for _, row in tqdm(
            self.data.iterrows(),
            total=len(self.data),
            desc="Processing dataset",
            disable=not verbose,
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

        return smiles, spec_names, formulas, instruments, nodes, edges

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
