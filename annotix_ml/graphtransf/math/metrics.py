import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator, rdRascalMCES


def MCES_distance(smile1: str, smile2: str) -> int:
    # Create the mol associated to the smiles
    mol1 = Chem.MolFromSmiles(smile1)
    mol2 = Chem.MolFromSmiles(smile2)

    # Compute the MCES of the two molecules
    mces = rdRascalMCES.FindMCES(mol1, mol2)
    if len(mces) == 0:
        Ec = 0

    else:
        Ec = len(mces[0].bondMatches())

    # Compute the distance
    Eone = mol1.GetNumBonds()
    Etwo = mol2.GetNumBonds()

    mces_distance = (Eone + Etwo) - (2 * Ec)

    return mces_distance


def mol_to_fingerprint(m: Chem.Mol, radius: int = 3, nbits: int = 2048):
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    curr_fp = morgan_gen.GetFingerprint(m)
    fingerprint = np.zeros((0,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(curr_fp, fingerprint)

    return fingerprint


def tanimoto_sim(smile1: str, smile2: str) -> float:
    # Create the mol associated to the smiles
    mol1 = Chem.MolFromSmiles(smile1)
    mol2 = Chem.MolFromSmiles(smile2)

    # Compute the fingerprints of the mol
    fp1 = mol_to_fingerprint(mol1)
    fp2 = mol_to_fingerprint(mol2)

    # Compute the tanimoto similarity
    intersection = (fp1 & fp2).sum(-1)
    union = (fp1 | fp2).sum(-1)
    tanimoto = intersection / union

    return tanimoto
