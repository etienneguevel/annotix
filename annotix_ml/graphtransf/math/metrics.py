import numpy as np
import torch
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


def compute_validity(pred_smiles: list[str | None]) -> list[float]:
    """
    Compute the validity of the predicted smiles.
    Returns a list of 1.0 for valid smiles and 0.0 for invalid ones.
    """
    return [1.0 if s is not None else 0.0 for s in pred_smiles]


def compute_tanimoto_similarity(
    pred_smiles: list[str | None], true_smiles: list[str | None]
) -> list[float]:
    """
    Compute the Tanimoto similarity for valid predicted smiles.
    Returns a list of similarities for valid predictions only.
    """
    tanimoto_sims = []
    for p_s, t_s in zip(pred_smiles, true_smiles):
        if p_s is not None and t_s is not None:
            try:
                sim = tanimoto_sim(p_s, t_s)
                tanimoto_sims.append(sim)
            except Exception:
                pass
    return tanimoto_sims


def compute_accuracy(
    pN: torch.Tensor,
    pE: torch.Tensor,
    N: torch.Tensor,
    E: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, list[float]]:
    """
    Compute the node and edge accuracy for each graph in the batch.
    Returns a dictionary with lists of accuracy values.
    """
    # Node accuracy
    N_target = N.argmax(-1)  # (bs, n)
    pN_pred = pN.argmax(-1)  # (bs, n)

    # Compare predictions
    node_correct = (N_target == pN_pred).float()

    # Mask invalid nodes
    mask_bool = mask > 0
    node_correct_sum = (node_correct * mask_bool).sum(dim=1)
    node_count = mask.sum(dim=1)
    node_accuracy = node_correct_sum / (node_count + 1e-8)

    # Edge accuracy
    E_target = E.argmax(-1)  # (bs, n, n)
    pE_pred = pE.argmax(-1)  # (bs, n, n)

    # Compare predictions
    edge_correct = (E_target == pE_pred).float()

    # Mask invalid edges
    mask_edges = mask.unsqueeze(1) * mask.unsqueeze(2)
    edge_correct_sum = (edge_correct * mask_edges).sum(dim=(1, 2))
    edge_count = mask_edges.sum(dim=(1, 2))
    edge_accuracy = edge_correct_sum / (edge_count + 1e-8)

    return {
        "node_accuracy": node_accuracy.tolist(),
        "edge_accuracy": edge_accuracy.tolist(),
    }


def compute_metrics(
    pred_smiles: list[str | None],
    true_smiles: list[str | None],
    pN: torch.Tensor,
    pE: torch.Tensor,
    N: torch.Tensor,
    E: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, list[float]]:
    """
    Compute the metrics for the evaluation loop.
    Returns a dictionary where values are lists of metrics.
    """
    metrics = {}

    metrics["validity"] = compute_validity(pred_smiles)
    metrics["tanimoto_similarity"] = compute_tanimoto_similarity(
        pred_smiles, true_smiles
    )

    accuracy_metrics = compute_accuracy(pN, pE, N, E, mask)
    metrics.update(accuracy_metrics)

    return metrics
