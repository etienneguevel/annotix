import numpy as np
import torch
import rdkit.Chem as Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator, rdRascalMCES


def MCES_distance(smile1: str, smile2: str) -> int:
    """
    Compute the Maximum Common Edge Subgraph (MCES) distance between two molecules.

    The distance is defined as: (num_bonds1 + num_bonds2) - (2 * num_mces_bonds).

    Args:
        smile1 (str): SMILES string of the first molecule.
        smile2 (str): SMILES string of the second molecule.

    Returns:
        int: The MCES distance between the two molecules.
    """
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
    """
    Convert a RDKit molecule object to a numpy Morgan fingerprint.

    Args:
        m (Chem.Mol): RDKit molecule object.
        radius (int, optional): Radius for Morgan fingerprint. Defaults to 3.
        nbits (int, optional): Number of bits in the fingerprint. Defaults to 2048.

    Returns:
        np.ndarray: Morgan fingerprint as a numpy array.
    """
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    curr_fp = morgan_gen.GetFingerprint(m)
    fingerprint = np.zeros((0,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(curr_fp, fingerprint)

    return fingerprint


def tanimoto_sim(smile1: str, smile2: str) -> float:
    """
    Compute Tanimoto similarity between two SMILES strings using Morgan fingerprints.

    Args:
        smile1 (str): SMILES string of the first molecule.
        smile2 (str): SMILES string of the second molecule.

    Returns:
        float: Tanimoto similarity score.
    """
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
    Compute the validity score for a list of predicted SMILES.

    Args:
        pred_smiles (list[str | None]): List of predicted SMILES strings (None if invalid).

    Returns:
        list[float]: List of validity indicators (1.0 for valid, 0.0 for invalid).
    """
    return [1.0 if s is not None else 0.0 for s in pred_smiles]


def compute_tanimoto_similarity(
    pred_smiles: list[str | None], true_smiles: list[str | None]
) -> list[float]:
    """
    Compute Tanimoto similarities for pairs of predicted and ground truth SMILES.

    Args:
        pred_smiles (list[str | None]): List of predicted SMILES strings.
        true_smiles (list[str | None]): List of ground truth SMILES strings.

    Returns:
        list[float]: List of similarity values for valid predicted/true pairs.
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
    Compute node and edge categorical accuracy for a batch of graphs.

    Args:
        pN (torch.Tensor): Predicted node probabilities of shape (bs, n, natoms).
        pE (torch.Tensor): Predicted edge probabilities of shape (bs, n, n, nbonds).
        N (torch.Tensor): Target node one-hot labels of shape (bs, n, natoms).
        E (torch.Tensor): Target edge one-hot labels of shape (bs, n, n, nbonds).
        mask (torch.Tensor): Node mask of shape (bs, n).

    Returns:
        dict[str, list[float]]: Dictionary containing 'node_accuracy' and 'edge_accuracy' lists.
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
    Aggregate various evaluation metrics for a batch.

    Metrics include validity, Tanimoto similarity, and node/edge accuracy.

    Args:
        pred_smiles (list[str | None]): List of predicted SMILES.
        true_smiles (list[str | None]): List of ground truth SMILES.
        pN (torch.Tensor): Predicted node probabilities.
        pE (torch.Tensor): Predicted edge probabilities.
        N (torch.Tensor): Target node features.
        E (torch.Tensor): Target edge features.
        mask (torch.Tensor): Node mask.

    Returns:
        dict[str, list[float]]: Dictionary of lists containing computed metrics for each sample.
    """
    metrics = {}

    metrics["validity"] = compute_validity(pred_smiles)
    metrics["tanimoto_similarity"] = compute_tanimoto_similarity(
        pred_smiles, true_smiles
    )

    accuracy_metrics = compute_accuracy(pN, pE, N, E, mask)
    metrics.update(accuracy_metrics)

    return metrics
