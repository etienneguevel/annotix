import h5py
import json
import logging
from typing import Dict, Callable
from abc import ABC, abstractmethod
from pathlib import Path


import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint

from annotix_ml import ROOT
from annotix_ml.spectraencoder.utils import (
    unpack_bits,
    ION_LST,
    formula_to_dense,
    VALID_MONO_MASSES,
    ion_to_idx,
    get_instr_idx,
)
from annotix_ml.spectraencoder.data.objects import Mol, Spectra

ATOM_DECODER = ["C", "O", "P", "N", "S", "Cl", "F", "H"]
TYPES = {atom: i for i, atom in enumerate(ATOM_DECODER)}
BONDS = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}


def get_mol_featurizer(mol_features, **kwargs):
    return {
        "fingerprint": FingerprintFeaturizer,
    }[mol_features](**kwargs)


def get_spec_featurizer(spec_features, **kwargs):
    return {
        "peakformula": PeakFormula,
    }[spec_features](**kwargs)


def get_paired_featurizer(spec_features, mol_features, **kwargs):
    """get_paired_featurizer.

    Args:
        spec_features (str): Spec featurizer
        mol_features (str): Mol featurizer

    """

    mol_featurizer = get_mol_featurizer(mol_features, **kwargs)
    spec_featurizer = get_spec_featurizer(spec_features, **kwargs)
    paired_featurizer = PairedFeaturizer(spec_featurizer, mol_featurizer, **kwargs)
    return paired_featurizer


class PairedFeaturizer(object):
    """PairedFeaturizer"""

    def __init__(self, spec_featurizer, mol_featurizer, graph_featurizer=None, **kwarg):
        """__init__."""
        self.spec_featurizer = spec_featurizer
        self.mol_featurizer = mol_featurizer
        self.graph_featurizer = graph_featurizer

    def featurize_mol(self, mol: Mol, **kwargs) -> Dict:
        return self.mol_featurizer.featurize(mol, **kwargs)

    def featurize_spec(self, mol: Mol, **kwargs) -> Dict:
        return self.spec_featurizer.featurize(mol, **kwargs)

    def featurize_graph(self, mol: Mol, **kwargs) -> Dict | None:
        if self.graph_featurizer is not None:
            return self.graph_featurizer.featurize(mol, **kwargs)
        else:
            return None

    def get_mol_collate(self) -> Callable:
        return self.mol_featurizer.collate_fn

    def get_spec_collate(self) -> Callable:
        return self.spec_featurizer.collate_fn

    def get_graph_collate(self) -> Callable | None:
        if self.graph_featurizer is not None:
            return self.graph_featurizer.collate_fn
        else:
            return None

    def set_spec_featurizer(self, spec_featurizer):
        self.spec_featurizer = spec_featurizer

    def set_mol_featurizer(self, mol_featurizer):
        self.mol_featurizer = mol_featurizer

    def set_graph_featurizer(self, graph_featurizer):
        self.graph_featurizer = graph_featurizer


class Featurizer(ABC):
    """Featurizer"""

    def __init__(self, cache_featurizers: bool = False, **kwargs):
        super().__init__()
        self.cache_featurizers = cache_featurizers
        self.cache = {}

    @abstractmethod
    def _encode(self, obj: object) -> str:
        """Encode object into a string representation"""
        raise NotImplementedError()

    def _featurize(self, obj: object) -> Dict:
        """Internal featurize class that does not utilize the cache"""
        raise dict()

    def featurize(self, obj: object, train_mode=False, **kwargs) -> Dict:
        """Featurizer a single object"""
        encoded_obj = self._encode(obj)

        if self.cache_featurizers:
            if encoded_obj in self.cache:
                featurized = self.cache[encoded_obj]
            else:
                featurized = self._featurize(obj)
                self.cache[encoded_obj] = featurized
        else:
            featurized = self._featurize(obj)

        return featurized


class MolFeaturizer(Featurizer):
    """MolFeaturizer"""

    def _encode(self, mol: Mol) -> str:
        """Encode mol into smiles repr"""
        smi = mol.get_smiles()
        return smi


class SpecFeaturizer(Featurizer):
    """SpecFeaturizer"""

    def _encode(self, spec: Spectra) -> str:
        """Encode spectra into name"""
        return spec.get_spec_name()


class GraphFeaturizer(Featurizer):
    """GraphFeaturizer"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.morgan_r = kwargs.get("morgan_r", 2)
        self.morgan_nbits = kwargs.get("morgan_nbits", 2048)
        self.morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.morgan_r, fpSize=self.morgan_nbits
        )

    def _encode(self, mol: Mol) -> str | None:
        """Encode graph into name"""
        return mol.inchikey

    @staticmethod
    def collate_fn(graphs: list[Data]) -> Batch:
        return Batch.from_data_list(graphs)

    def _featurize(self, obj: Mol) -> Data:
        mol = Chem.MolFromSmiles(obj.get_smiles())
        smi = Chem.MolToSmiles(mol, isomericSmiles=False)
        mol = Chem.MolFromSmiles(smi)

        N = mol.GetNumAtoms()

        type_idx = []
        for atom in mol.GetAtoms():
            type_idx.append(TYPES[atom.GetSymbol()])

        row, col, edge_type = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_type += 2 * [
                BONDS[bond.GetBondType()] + 1
            ]  # add one so that 0 is reserved for no edge

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_type = torch.tensor(edge_type, dtype=torch.long)
        edge_attr = F.one_hot(edge_type, num_classes=len(BONDS) + 1).to(torch.float)

        permutation = (
            edge_index[0] * N + edge_index[1]
        ).argsort()  # sort by row then by column index
        edge_index = edge_index[:, permutation]
        edge_attr = edge_attr[permutation]

        x = F.one_hot(torch.tensor(type_idx), num_classes=len(TYPES)).float()
        y = torch.tensor(
            np.asarray(self.morgan_gen.GetFingerprint(mol), dtype=np.int8)
        ).unsqueeze(0)

        inchi = Chem.MolToInchi(mol)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, inchi=inchi)

        return data


class FingerprintFeaturizer(MolFeaturizer):
    """MolFeaturizer"""

    def __init__(self, fp_names: list[str], fp_file: str | None = None, **kwargs):
        """__init__

        Args:
            fp_names (list[str]): list of
            nbits (int): Number of bits
            fp_file (str): Saved fp file

        """
        super().__init__(**kwargs)
        self._fp_cache = {}
        self._morgan_projection = np.random.randn(50, 2048)
        self.fp_names = fp_names
        self.fp_file = fp_file

        # Only for csi fp
        self._root_dir = Path().resolve()

    def __getstate__(self):
        """Remove unpicklable h5py objects before pickling"""
        state = self.__dict__.copy()
        # Remove h5py file handles from cache as they can't be pickled
        cleaned_cache = {}
        for key, value in state.get("_fp_cache", {}).items():
            if isinstance(value, dict) and "features" in value:
                # Don't include the h5py dataset
                cleaned_cache[key] = {k: v for k, v in value.items() if k != "features"}
            else:
                cleaned_cache[key] = value
        state["_fp_cache"] = cleaned_cache
        return state

    def __setstate__(self, state):
        """Restore state after unpickling"""
        self.__dict__.update(state)

    @staticmethod
    def collate_fn(mols: list[dict]) -> dict:
        fp_ar = torch.tensor(np.array(mols))
        return {"mols": fp_ar}

    def featurize_smiles(self, smiles: str) -> np.ndarray:
        """featurize_smiles.

        Args:
            smiles (str): smiles
            kwargs:

        Returns:
            Dict:
        """

        mol_obj = Mol.MolFromSmiles(smiles)
        return self._featurize(mol_obj)

    def _featurize(self, mol: Mol):
        """featurize.

        Args:
            mol (Mol)

        """
        fp_list = []
        for fp_name in self.fp_names:
            # Get all fingerprint bits
            fingerprint = self._get_fingerprint(mol, fp_name)
            fp_list.append(fingerprint)

        fp = np.concatenate(fp_list)
        return fp

    # Fingerprint functions
    def _get_morgan_fp_base(self, mol: Mol, nbits: int = 2048, radius=2):
        """get morgan fingeprprint"""

        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=nbits
        )

        def fp_fn(m):
            return morgan_gen.GetFingerprint(m)

        mol = mol.get_rdkit_mol()
        fingerprint = fp_fn(mol)
        array = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        return array

    def _get_morgan_2048(self, mol: Mol):
        """get morgan fingeprprint"""
        return self._get_morgan_fp_base(mol, nbits=2048)

    def _get_morgan_projection(self, mol: Mol):
        """get morgan fingeprprint"""

        morgan_fp = self._get_morgan_fp_base(mol, nbits=2048)

        output_fp = np.einsum("ij,j->i", self._morgan_projection, morgan_fp)
        return output_fp

    def _get_morgan_1024(self, mol: Mol):
        """get morgan fingeprprint"""
        return self._get_morgan_fp_base(mol, nbits=1024)

    def _get_morgan_512(self, mol: Mol):
        """get morgan fingeprprint"""
        return self._get_morgan_fp_base(mol, nbits=512)

    def _get_morgan_256(self, mol: Mol):
        """get morgan fingeprprint"""
        return self._get_morgan_fp_base(mol, nbits=256)

    def _get_morgan_4096(self, mol: Mol):
        """get morgan fingeprprint"""
        return self._get_morgan_fp_base(mol, nbits=4096)

    def _get_morgan_4096_3(self, mol: Mol):
        """get morgan fingeprprint"""
        return self._get_morgan_fp_base(mol, nbits=4096, radius=3)

    def _get_maccs(self, mol: Mol):
        """get maccs fingerprint"""
        mol = mol.get_rdkit_mol()
        fingerprint = GetMACCSKeysFingerprint(mol)
        array = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        return array

    def _fill_precomputed_cache_hdf5(self, fp_file):
        """Get precomputed fp cache"""
        if fp_file not in self._fp_cache:
            if not Path(fp_file).exists():
                raise ValueError(f"Cannot find file {fp_file}")

            # Then get hdf5
            logging.info("Loading h5 features")
            dataset = h5py.File(fp_file, "r")
            logging.info("Stored in fp_cache")
            index = {
                i.decode(): ind for ind, i in enumerate(np.array(dataset["ikeys"]))
            }
            num_bits = dataset.attrs["num_bits"]
            self._fp_cache[fp_file] = {
                "index": index,
                "features": dataset["features"],
                "num_bits": num_bits,
            }

    def _get_precomputed_hdf5(self, mol, fp_file):
        """Get precomputed hdf5 of a single molecule"""
        self._fill_precomputed_cache_hdf5(fp_file)
        cache_obj = self._fp_cache[fp_file]
        index = cache_obj["index"]
        feats = cache_obj["features"]
        inchikey = mol.get_inchikey()
        if inchikey in index:
            num_bits = cache_obj["num_bits"]
            out_vec = unpack_bits(feats[index[inchikey]], num_bits=num_bits)
            return out_vec
        else:
            num_bits = cache_obj["num_bits"]
            logging.info(f"Unable to find inchikey {inchikey} in {fp_file}")
            # Create empty vector
            return np.zeros(num_bits)

    def _get_csi(self, mol):
        return self._get_precomputed_hdf5(mol, self.fp_file)

    @classmethod
    def get_fingerprint_size(cls, fp_names: list = []):
        """Get list of fingerprint size"""
        fp_name_to_bits = {
            "morgan256": 256,
            "morgan512": 512,
            "morgan1024": 1024,
            "morgan2048": 2048,
            "morgan_project": 50,
            "morgan4096": 4096,
            "morgan4096_3": 4096,
            "maccs": 167,
            "csi": 5496,
        }
        num_bits = 0
        for fp_name in fp_names:
            num_bits += fp_name_to_bits.get(fp_name)
        return num_bits

    def _get_fingerprint(self, mol: Mol, fp_name: str):
        """_get_fingerprint_fn"""
        return {
            "morgan256": self._get_morgan_256,
            "morgan512": self._get_morgan_512,
            "morgan1024": self._get_morgan_1024,
            "morgan2048": self._get_morgan_2048,
            "morgan_project": self._get_morgan_projection,
            "morgan4096": self._get_morgan_4096,
            "morgan4096_3": self._get_morgan_4096_3,
            "maccs": self._get_maccs,
            "csi": self._get_csi,
        }[fp_name](mol)

    def dist(self, mol_1, mol_2) -> np.ndarray:
        """Return 2048 bit molecular fingerprint"""
        fp1 = self.featurize(mol_1)
        fp2 = self.featurize(mol_2)
        tani = 1 - (((fp1 & fp2).sum()) / (fp1 | fp2).sum())
        return tani

    def dist_batch(self, mol_list) -> np.ndarray:
        """Return 2048 bit molecular fingerprint"""

        fps = []
        if len(mol_list) == 0:
            return np.array([[]])

        for mol_temp in mol_list:
            fps.append(self.featurize(mol_temp))

        fps = np.vstack(fps)

        fps_a = fps[:, None, :]
        fps_b = fps[None, :, :]

        intersect = (fps_a & fps_b).sum(-1)
        union = (fps_a | fps_b).sum(-1)
        tani = 1 - intersect / union
        return tani

    def dist_one_to_many(self, mol, mol_list) -> np.ndarray:
        """Return 2048 bit molecular fingerprint"""

        fps = []
        if len(mol_list) == 0:
            return np.array([[]])

        for mol_temp in mol_list:
            fps.append(self.featurize(mol_temp))

        fp_a = self.featurize(mol)

        fps = np.vstack(fps)

        fps_a = fp_a[None, :]
        fps_b = fps

        intersect = (fps_a & fps_b).sum(-1)
        union = (fps_a | fps_b).sum(-1)

        # Compute dist
        tani = 1 - intersect / union
        return tani


class PeakFormula(SpecFeaturizer):
    """PeakFormula."""

    cat_types = {"frags": 0, "loss": 1, "ab_loss": 2, "cls": 3}
    num_inten_bins = 10
    num_types = len(cat_types)
    cls_type = cat_types.get("cls")

    num_adducts = len(ION_LST)

    def __init__(
        self,
        subform_folder: str,
        forward_labels: str | None = None,
        augment_data: bool = False,
        augment_prob: float = 1,
        remove_prob: float = 0.1,
        remove_weights: float = "uniform",
        inten_prob: float = 0.1,
        cls_type: str = "ms1",
        magma_aux_loss: bool = False,
        magma_folder: str | None = None,
        forward_aug_folder: str | None = None,
        max_peaks: int | None = None,
        inten_transform: str = "float",
        magma_modulo: int = 512,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cls_type = cls_type
        self.forward_labels = forward_labels
        self.augment_data = augment_data
        self.remove_prob = remove_prob
        self.augment_prob = augment_prob
        self.remove_weights = remove_weights
        self.inten_prob = inten_prob
        self.magma_aux_loss = magma_aux_loss
        self.forward_aug_folder = forward_aug_folder
        self.max_peaks = max_peaks
        self.inten_transform = inten_transform
        self.aug_nbits = magma_modulo

        self.spec_name_to_subform_file = {
            i.stem: i for i in list((ROOT / subform_folder).glob("*.json"))
        }

        if self.forward_labels is not None and self.forward_aug_folder is not None:
            self.forward_aug_folder = ROOT / self.forward_aug_folder
            self.spec_name_to_subform_file.update(
                {i.stem: i for i in self.forward_aug_folder.glob("*.json")}
            )

        self.spec_name_to_magma_file = {}
        if self.magma_aux_loss:
            if magma_folder is None:
                raise ValueError(
                    "magma_folder must be specified if magma_aux_loss is True"
                )

            self.magma_folder = ROOT / magma_folder
            name_map = {
                i: self.magma_folder / f"{i}.magma"
                for i in self.spec_name_to_subform_file.keys()
            }
            self.spec_name_to_magma_file = {
                k: v for k, v in name_map.items() if v.exists()
            }

    def _get_peak_dict(self, spec: Spectra) -> dict:
        """_get_peak_dict.

        Args:
            spec (Spectra): spec

        Returns:
            dict: information about the specter
                - frags
                - intens
                - ions
                - root_form
                - root_ion
        """

        spec_name = self._encode(spec)

        # Default empty peak dict structure
        empty_peak_dict = {
            "frags": [],
            "intens": [],
            "ions": [],
            "root_form": spec.get_spectra_formula()
            if spec.get_spectra_formula()
            else "C",
            "root_ion": "[M+H]+",
        }

        # Handle case where subform_file is not in dictionary
        if spec_name not in self.spec_name_to_subform_file:
            return empty_peak_dict

        subform_file = Path(self.spec_name_to_subform_file[spec_name])

        if not subform_file.exists():
            return empty_peak_dict

        with open(subform_file, "r") as fp:
            tree = json.load(fp)

        root_form = tree["cand_form"]
        root_ion = tree["cand_ion"]
        output_tbl = tree["output_tbl"]

        if output_tbl is None:
            frags = []
            intens = []
            ions = []
        else:
            frags = output_tbl["formula"]
            intens = output_tbl["ms2_inten"]
            ions = output_tbl["ions"]

        out_dict = {
            "frags": frags,
            "intens": intens,
            "ions": ions,
            "root_form": root_form,
            "root_ion": root_ion,
        }

        # If we have a max peaks, then we need to filter
        if self.max_peaks is not None:
            # Sort by intensity
            inten_list = list(out_dict["intens"])

            new_order = np.argsort(inten_list)[::-1]
            cutoff_ind = min(len(inten_list) - 1, self.max_peaks)
            new_inds = new_order[:cutoff_ind]

            # Get new frags, intens, ions and assign to outdict
            inten_list = np.array(inten_list)[new_inds].tolist()
            frag_list = np.array(out_dict["frags"])[new_inds].tolist()
            ion_list = np.array(out_dict["ions"])[new_inds].tolist()

            out_dict["frags"] = frag_list
            out_dict["intens"] = inten_list
            out_dict["ions"] = ion_list

        return out_dict

    def augment_peak_dict(self, peak_dict: dict, **kwargs):
        """augment_peak_dict.

        Add peaks, remove, peaks, or rescale peaks

        Args:
            peak_dict (dict): Dictionary containing peak dict info to augment

        Return:
            peak_dict
        """

        # Only scale frags
        frags = np.array(peak_dict["frags"])
        intens = np.array(peak_dict["intens"])
        ions = np.array(peak_dict["ions"])

        # Compute removal probability
        num_modify_peaks = len(frags)  # - 1
        keep_prob = 1 - self.remove_prob
        num_to_keep = np.random.binomial(
            n=num_modify_peaks,
            p=keep_prob,
        )

        if len(frags) == 0:
            return peak_dict
        # Temp
        keep_inds = np.arange(0, num_modify_peaks)  # + 1)

        # Quadratic probability weighting
        if self.remove_weights == "quadratic":
            keep_probs = intens[0:].reshape(-1) ** 2 + 1e-9
            keep_probs = keep_probs / keep_probs.sum()
        elif self.remove_weights == "uniform":
            keep_probs = intens[0:] + 1e-9
            keep_probs = np.ones(len(keep_probs)) / len(keep_probs)
        elif self.remove_weights == "exp":
            # Temp
            # keep_probs = start_intens[1:] + 1e-9
            keep_probs = np.exp(intens[0:].reshape(-1) + 1e-5)
            keep_probs = keep_probs / keep_probs.sum()
        else:
            raise NotImplementedError()

        # Keep indices
        # Add root
        ind_samples = np.random.choice(
            keep_inds, size=num_to_keep, replace=False, p=keep_probs
        )
        # Re-index frags, intens, and ions
        frags, intens, ions = frags[ind_samples], intens[ind_samples], ions[ind_samples]

        rescale_prob = np.random.random(len(intens))
        inten_scalar_factor = np.random.normal(loc=1, size=len(intens))
        inten_scalar_factor[inten_scalar_factor <= 0] = 0

        # Where rescale prob is >= self.inten_prob set inten rescale to 1
        inten_scalar_factor[rescale_prob >= self.inten_prob] = 1

        # Rescale intens
        intens = intens * inten_scalar_factor
        new_max = intens.max() + 1e-12 if len(intens) > 0 else 1
        intens /= new_max
        # Replace peak dict with new values
        peak_dict["intens"] = intens
        peak_dict["frags"] = frags
        peak_dict["ions"] = ions

        return peak_dict

    def _featurize(self, spec: Spectra, train_mode: bool = False, **kwargs) -> Dict:
        """featurize.

        Args:
            spec (Spectra)

        """
        spec_name = spec.get_spec_name()

        # Return get_peak_formulas output
        peak_dict = self._get_peak_dict(spec)

        # Augment peak dict with chem formulae
        if train_mode and self.augment_data:
            # Only augment certain select peaks
            augment_peak = np.random.random() < self.augment_prob
            if augment_peak:
                peak_dict = self.augment_peak_dict(peak_dict)

        # Add in chemical formuale
        root = peak_dict["root_form"]

        forms_vec = [formula_to_dense(i) for i in peak_dict["frags"]]
        if len(forms_vec) == 0:
            mz_vec = []
        else:
            mz_vec = (np.array(forms_vec) * VALID_MONO_MASSES).sum(-1).tolist()
        root_vec = formula_to_dense(root)
        root_ion = ion_to_idx.get(peak_dict["root_ion"])
        root_mass = (root_vec * VALID_MONO_MASSES).sum()
        inten_vec = list(peak_dict["intens"])
        ion_vec = [ion_to_idx.get(i) for i in peak_dict["ions"]]
        type_vec = len(forms_vec) * [self.cat_types["frags"]]
        instrument = get_instr_idx(spec.get_instrument())

        if self.cls_type == "ms1":
            cls_ind = self.cat_types.get("cls")
            inten_vec.append(1.0)
            type_vec.append(cls_ind)
            forms_vec.append(root_vec)
            mz_vec.append(root_mass)
            ion_vec.append(root_ion)

        elif self.cls_type == "zeros":
            cls_ind = self.cat_types.get("cls")
            inten_vec.append(0.0)
            type_vec.append(cls_ind)
            forms_vec.append(np.zeros_like(root_vec))
            mz_vec.append(0)
            ion_vec.append(root_ion)
        else:
            raise NotImplementedError()

        # Featurize all formulae
        inten_vec = np.array(inten_vec)
        if self.inten_transform == "float":
            self.inten_feats = 1
        elif self.inten_transform == "zero":
            self.inten_feats = 1
            inten_vec = np.zeros_like(inten_vec)
        elif self.inten_transform == "log":
            self.inten_feats = 1
            inten_vec = np.log(inten_vec + 1e-5)
        elif self.inten_transform == "cat":
            self.inten_feats = self.num_inten_bins
            bins = np.linspace(0, 1, self.num_inten_bins)
            # Digitize inten vec
            inten_vec = np.digitize(inten_vec, bins)
        else:
            raise NotImplementedError()

        forms_vec = np.array(forms_vec)

        # Add in magma supervision!
        magma_file = self.spec_name_to_magma_file.get(spec_name)
        fingerprints = np.zeros((forms_vec.shape[0], self.aug_nbits)) - 1
        if self.magma_aux_loss and magma_file is not None:
            magma_df = pd.read_csv(magma_file, sep="\t")
            if len(magma_df) > 0:
                mz_vec = np.array(mz_vec)
                magma_masses = magma_df["mz_corrected"].values

                # Get closest mz within 1e-4
                diff_mat = np.abs(mz_vec[:, None] - magma_masses[None, :])
                min_inds = diff_mat.argmin(1)
                for row_ind, (mz_val, min_ind) in enumerate(zip(mz_vec, min_inds)):
                    diff_val = diff_mat[row_ind, min_ind]
                    if diff_val < 1e-4:
                        # Get fp!
                        magma_row = magma_df.iloc[min_ind]
                        magma_fp_bits = [
                            int(i) % self.aug_nbits
                            for i in magma_row["frag_fp"].split(",")
                        ]

                        # Set to base of 0
                        fingerprints[row_ind, :] = 0

                        # Update to 1 where active bit
                        fingerprints[row_ind, magma_fp_bits] = 1

        # Use int featurizer and norm later
        out_dict = {
            "peak_type": np.array(type_vec),
            "form_vec": forms_vec,
            "ion_vec": ion_vec,
            "frag_intens": inten_vec,
            "name": spec_name,
            "magma_fps": fingerprints,
            "magma_aux_loss": self.magma_aux_loss,
            "instrument": instrument,
        }
        return out_dict

    @classmethod
    def get_num_inten_feats(cls, inten_transform):
        """_summary_

        Args:
            inten_transform (_type_): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """
        if inten_transform == "float":
            inten_feats = 1
        elif inten_transform == "zero":
            inten_feats = 1
        elif inten_transform == "log":
            inten_feats = 1
        elif inten_transform == "cat":
            inten_feats = PeakFormula.num_inten_bins
        else:
            raise NotImplementedError()
        return inten_feats

    def _extract_fingerprint(self, smiles):
        """extract_fingerprints."""
        index = self.fp_index_obj.get(smiles)
        return self.fp_dataset[index]

    def featurize(self, spec: Spectra, train_mode=False, **kwargs) -> Dict:
        """Featurizer a single object"""

        encoded_obj = self._encode(spec)
        if train_mode:
            featurized = self._featurize(spec, train_mode=train_mode)
        else:
            if self.cache_featurizers:
                if encoded_obj in self.cache:
                    featurized = self.cache[encoded_obj]
                else:
                    featurized = self._featurize(spec)
                    self.cache[encoded_obj] = featurized
            else:
                featurized = self._featurize(spec)

        return featurized

    @staticmethod
    def collate_fn(input_list: list[dict]) -> Dict:
        """_summary_

        Args:
            input_list (list[dict]): _description_

        Returns:
            Dict: _description_
        """
        # Determines the number of channels
        names = [j["name"] for j in input_list]
        peak_form_tensors = [torch.from_numpy(j["form_vec"]) for j in input_list]
        inten_tensors = [torch.from_numpy(j["frag_intens"]) for j in input_list]
        type_tensors = [torch.from_numpy(j["peak_type"]) for j in input_list]
        instrument_tensors = torch.FloatTensor([j["instrument"] for j in input_list])
        ion_tensors = [torch.FloatTensor(j["ion_vec"]) for j in input_list]

        peak_form_lens = np.array([i.shape[0] for i in peak_form_tensors])
        max_len = np.max(peak_form_lens)
        padding_amts = max_len - peak_form_lens

        type_tensors = [
            torch.nn.functional.pad(i, (0, pad_len))
            for i, pad_len in zip(type_tensors, padding_amts)
        ]
        ion_tensors = [
            torch.nn.functional.pad(i, (0, pad_len))
            for i, pad_len in zip(ion_tensors, padding_amts)
        ]
        inten_tensors = [
            torch.nn.functional.pad(i, (0, pad_len))
            for i, pad_len in zip(inten_tensors, padding_amts)
        ]
        peak_form_tensors = [
            torch.nn.functional.pad(i, (0, 0, 0, pad_len))
            for i, pad_len in zip(peak_form_tensors, padding_amts)
        ]

        # Stack everything (bxd for root, bxp for others)
        type_tensors = torch.stack(type_tensors, dim=0).long()
        peak_form_tensors = torch.stack(peak_form_tensors, dim=0).float()
        ion_tensors = torch.stack(ion_tensors, dim=0).float()

        inten_tensors = torch.stack(inten_tensors, dim=0).float()
        num_peaks = torch.from_numpy(peak_form_lens).long()

        # magma_fps
        use_magma = np.any([i["magma_aux_loss"] for i in input_list])
        magma_dict = {}
        if use_magma:
            magma_fingerprints = [i["magma_fps"] for i in input_list]

            # fingerprints: Batch x max num peaks x fingerprint dimension
            for i in range(len(magma_fingerprints)):
                padded_fp = np.zeros((max_len, magma_fingerprints[0].shape[1]))
                padded_fp[: magma_fingerprints[i].shape[0], :] = magma_fingerprints[i]
                magma_fingerprints[i] = padded_fp

            magma_fingerprints = np.stack(magma_fingerprints, axis=0)
            magma_fingerprints = torch.tensor(magma_fingerprints, dtype=torch.float)
            magma_dict["fingerprints"] = magma_fingerprints

            # Mask for where the spectra doesn't have a peak or has a peak but not a fingerprint
            fingerprint_sum = magma_fingerprints.sum(2)
            fingerprint_mask = fingerprint_sum > 0
            magma_dict["fingerprint_mask"] = fingerprint_mask

        return_dict = {
            "types": type_tensors,
            "form_vec": peak_form_tensors,
            "ion_vec": ion_tensors,
            "intens": inten_tensors,
            "names": names,
            "num_peaks": num_peaks,
            "instruments": instrument_tensors,
        }

        return_dict.update(magma_dict)
        return return_dict
