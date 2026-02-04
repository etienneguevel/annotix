from collections import defaultdict
from itertools import groupby
from pathlib import Path

import numpy as np
import re
import torch
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from torch_geometric.utils import to_dense_batch, to_dense_adj, remove_self_loops


CHEM_FORMULA_SIZE = "([A-Z][a-z]*)([0-9]*)"

ION_LST = [
    "[M+H]+",
    "[M+Na]+",
    "[M+K]+",
    "[M-H2O+H]+",
    "[M+H3N+H]+",
    "[M]+",
    "[M-H4O2+H]+",
]

VALID_ELEMENTS = [
    "C",
    "H",
    "As",
    "B",
    "Br",
    "Cl",
    "Co",
    "F",
    "Fe",
    "I",
    "K",
    "N",
    "Na",
    "O",
    "P",
    "S",
    "Se",
    "Si",
]

ELEMENT_VECTORS = np.eye(len(VALID_ELEMENTS))
element_to_position = dict(zip(VALID_ELEMENTS, ELEMENT_VECTORS))

P_TBL = Chem.GetPeriodicTable()

VALID_MONO_MASSES = np.array(
    [P_TBL.GetMostCommonIsotopeMass(i) for i in VALID_ELEMENTS]
)

CHEM_MASSES = VALID_MONO_MASSES[:, None]
ELECTRON_MASS = 0.00054858
ELEMENT_TO_MASS = dict(zip(VALID_ELEMENTS, CHEM_MASSES.squeeze()))


# Reasonable normalization vector for elements
# Estimated by max counts (+ 1 when zero)
NORM_VEC = np.array([81, 158, 2, 1, 3, 10, 1, 17, 1, 6, 1, 19, 2, 34, 6, 6, 2, 6])

ion_to_idx = dict(zip(ION_LST, np.arange(len(ION_LST))))

ion_to_mass = {
    "[M+H]+": ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    "[M+Na]+": ELEMENT_TO_MASS["Na"] - ELECTRON_MASS,
    "[M+K]+": ELEMENT_TO_MASS["K"] - ELECTRON_MASS,
    "[M-H2O+H]+": -ELEMENT_TO_MASS["O"] - ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    "[M+H3N+H]+": ELEMENT_TO_MASS["N"] + ELEMENT_TO_MASS["H"] * 4 - ELECTRON_MASS,
    "[M]+": 0 - ELECTRON_MASS,
    "[M-H4O2+H]+": -ELEMENT_TO_MASS["O"] * 2 - ELEMENT_TO_MASS["H"] * 3 - ELECTRON_MASS,
}

instrument_to_type = defaultdict(lambda: "unknown")
instrument_to_type.update(
    {
        "Thermo Finnigan Velos Orbitrap": "orbitrap",
        "Thermo Finnigan Elite Orbitrap": "orbitrap",
        "Orbitrap Fusion Lumos": "orbitrap",
        "Q-ToF (LCMS)": "qtof",
        "Unknown (LCMS)": "unknown",
        "ion trap": "iontrap",
        "FTICR (LCMS)": "fticr",
        "Bruker Q-ToF (LCMS)": "qtof",
        "Orbitrap (LCMS)": "orbitrap",
        "Orbitrap": "orbitrap",
    }
)

instruments = sorted(list(set(instrument_to_type.values())))
max_instr_idx = len(instruments) + 1
instrument_to_idx = dict(zip(instruments, np.arange(len(instruments))))


def get_instr_idx(instrument: str) -> int:
    """map instrument to its index in one hot encoding"""
    inst = instrument_to_type.get(instrument, "unknown")
    return instrument_to_idx[inst]


def unpack_bits(vec, num_bits):
    return np.unpackbits(vec, axis=-1)[..., -num_bits:]


def formula_to_dense(chem_formula: str) -> np.ndarray:
    """formula_to_dense.

    Args:
        chem_formula (str): Input chemical formal
    Return:
        np.ndarray of vector

    """
    total_onehot = []
    for chem_symbol, num in re.findall(CHEM_FORMULA_SIZE, chem_formula):
        # Convert num to int
        num = 1 if num == "" else int(num)
        one_hot = element_to_position[chem_symbol].reshape(1, -1)
        one_hot_repeats = np.repeat(one_hot, repeats=num, axis=0)
        total_onehot.append(one_hot_repeats)

    # Check if null
    if len(total_onehot) == 0:
        dense_vec = np.zeros(len(element_to_position))
    else:
        dense_vec = np.vstack(total_onehot).sum(0)

    return dense_vec


class PlaceHolder:
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor):
        """Changes the device and dtype of X, E, y."""
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = -1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = -1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self


def parse_spectra(spectra_file: str) -> tuple[dict, list[tuple[str, np.ndarray]]]:
    """parse_spectra.

    Parses spectra in the SIRIUS format and returns

    Args:
        spectra_file (str): Name of spectra file to parse
    Return:
        Tuple[dict, List[Tuple[str, np.ndarray]]]: metadata and list of spectra
            tuples containing name and array
    """
    lines = [i.strip() for i in open(spectra_file, "r").readlines()]

    group_num = 0
    metadata = {}
    spectras = []
    my_iterator = groupby(
        lines, lambda line: line.startswith(">") or line.startswith("#")
    )

    for index, (start_line, lines) in enumerate(my_iterator):
        group_lines = list(lines)
        subject_lines = list(next(my_iterator)[1])
        # Get spectra
        if group_num > 0:
            spectra_header = group_lines[0].split(">")[1]
            peak_data = [
                [float(x) for x in peak.split()[:2]]
                for peak in subject_lines
                if peak.strip()
            ]
            # Check if spectra is empty
            if len(peak_data):
                peak_data = np.vstack(peak_data)
                # Add new tuple
                spectras.append((spectra_header, peak_data))
        # Get meta data
        else:
            entries = {}
            orphaned_header = None
            for i in group_lines:
                if " " not in i:
                    if i.startswith(">"):
                        orphaned_header = i
                    continue
                elif i.startswith("#INSTRUMENT TYPE"):
                    key = "#INSTRUMENT TYPE"
                    val = i.split(key)[1].strip()
                    entries[key[1:]] = val
                else:
                    start, end = i.split(" ", 1)
                    start = start[1:]
                    while start in entries:
                        start = f"{start}'"
                    entries[start] = end

            metadata.update(entries)

            if orphaned_header:
                spectra_header = orphaned_header.split(">")[1]
                peak_data = [
                    [float(x) for x in peak.split()[:2]]
                    for peak in subject_lines
                    if peak.strip()
                ]
                # Check if spectra is empty
                if len(peak_data):
                    peak_data = np.vstack(peak_data)
                    # Add new tuple
                    spectras.append((spectra_header, peak_data))
        group_num += 1

    metadata["_FILE_PATH"] = spectra_file
    metadata["_FILE"] = Path(spectra_file).stem
    return metadata, spectras


def uncharged_formula(mol, mol_type="mol") -> str:
    """Compute uncharged formula"""
    if mol_type == "mol":
        chem_formula = CalcMolFormula(mol)
    elif mol_type == "smiles":
        mol = Chem.MolFromSmiles(mol)
        if mol is None:
            return None
        chem_formula = CalcMolFormula(mol)
    else:
        raise ValueError()

    return re.findall(r"^([^\+,^\-]*)", chem_formula)[0]


def encode_no_edge(E):
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = (
        torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    )
    E[diag] = 0
    return E


def to_dense(x, edge_index, edge_attr, batch):
    X, node_mask = to_dense_batch(x=x, batch=batch)
    # node_mask = node_mask.float()
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    # TODO: carefully check if setting node_mask as a bool breaks the continuous case
    max_num_nodes = X.size(1)
    E = to_dense_adj(
        edge_index=edge_index,
        batch=batch,
        edge_attr=edge_attr,
        max_num_nodes=max_num_nodes,
    )
    E = encode_no_edge(E)

    return PlaceHolder(X=X, E=E, y=None), node_mask
