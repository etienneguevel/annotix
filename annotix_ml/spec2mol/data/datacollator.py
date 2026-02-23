from typing import List
import torch
from annotix_ml.spectraencoder.data.featurizers import PeakFormula


def graph_spec_collate_fn(batch: List[dict]) -> tuple:
    """
    Collate function for GraphSpecDataset.
    Handles graph padding and spectra feature collation.

    Returns:
        tuple: (nodes, edges, mask, num_peaks, types, instruments, ion_vec, form_vec, intens, smiles)
    """
    # Collate spectra features using PeakFormula.collate_fn logic
    spec_batch = PeakFormula.collate_fn(batch)

    # Collate graph features
    # Since nodes and edges have different sizes (N), we need padding
    node_tensors = [d["nodes"] for d in batch]
    edge_tensors = [d["edges"] for d in batch]

    num_nodes = torch.tensor([n.shape[0] for n in node_tensors])
    max_n = num_nodes.max()

    padded_nodes = []
    padded_edges = []
    node_masks = []

    for n, e in zip(node_tensors, edge_tensors):
        curr_n = n.shape[0]
        # Nodes: (n, natoms) -> (max_n, natoms)
        padded_n = torch.nn.functional.pad(n, (0, 0, 0, max_n - curr_n))
        padded_nodes.append(padded_n)

        # Edges: (n, n, nbonds) -> (max_n, max_n, nbonds)
        padded_e = torch.nn.functional.pad(
            e, (0, 0, 0, max_n - curr_n, 0, max_n - curr_n)
        )
        # Set NoBond (index 0) for padded areas
        padded_e[curr_n:, :, 0] = 1
        padded_e[:, curr_n:, 0] = 1
        padded_edges.append(padded_e)

        mask = torch.zeros(max_n, dtype=torch.bool)
        mask[:curr_n] = True
        node_masks.append(mask)

    nodes = torch.stack(padded_nodes).to(torch.float32)
    edges = torch.stack(padded_edges).to(torch.float32)
    mask = torch.stack(node_masks)
    smiles = [d["smiles"] for d in batch]

    return (
        nodes,
        edges,
        mask,
        spec_batch["num_peaks"],
        spec_batch["types"],
        spec_batch["instruments"],
        spec_batch["ion_vec"],
        spec_batch["form_vec"],
        spec_batch["intens"],
        smiles,
    )
