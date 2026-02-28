import torch
from torch.nn.attention.flex_attention import create_block_mask
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence

from annotix_ml.spectraencoder.data.featurizers import PeakFormula


def collateGraph(
    batch: list[tuple[torch.Tensor]],
):
    """
    Data collate function to transform the outputs of the dataset into batches
    that will be fed to the DataLoader objects during the training.
    This function is made to be used with the GraphDatasetFromSMILEs object and
    plug it to a torch DataLoader.

    Args:
    batch, which is a list of size 2 tuples made of :
    - N: torch.Tensor, contains node of size (n_mol, natoms)
    - E: torch.Tensor, contains edge of size (n_mol, n_mol, nbonds)

    Returns:
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
    - nodes: torch.Tensor of shape (bs, n_batch, natoms)
    - edges: torch.Tensor of shape (bs, n_batch, n_batch, nbonds)
    - mask: torch.Tensor of shape (bs, n_batch)
    """
    # Unpack the nodes, edges and pos_emb
    # bs = len(batch)
    list_N, list_E = zip(*batch)

    # Look at the number of atoms in each graph
    num_atoms = [N.shape[0] for N in list_N]
    n_batch = max(num_atoms)

    # Pad the nodes
    N_padded = (
        pad_sequence(list_N).transpose(0, 1).to(torch.float32)
    )  # (bs, n_batch, natoms)

    # Pad the edges
    E_padded = torch.cat(
        [
            pad(E, (0, 0, 0, n_batch - n, 0, n_batch - n)).unsqueeze(0)
            for n, E in zip(num_atoms, list_E)
        ],
        dim=0,
    ).to(torch.float32)  # (bs, n_batch, n_batch, k)

    # Make the mask
    mask = torch.stack(
        [torch.cat([torch.ones(m), torch.zeros(n_batch - m)]) for m in num_atoms]
    )  # (bs, n_batch)

    return N_padded, E_padded, mask


def collateGraphJagged(batch: list[tuple[torch.Tensor]]):
    # Collate the nodes and edges in a jagged way (nested tensors).
    # We will end with a dimensions :
    # - L -> total number of nodes of the batch
    # - M -> total number of edges of the batch
    list_N, list_E = zip(*batch)

    # Look at the number of atoms in each graph
    graph_id = torch.hstack(
        [torch.tensor([i for _ in range(N.shape[0])]) for i, N in enumerate(list_N)]
    )  # (L,)
    num_tokens = graph_id.shape[0]

    # Make the block mask
    def causal(b, h, q_idx, kv_idx):
        q_bs = graph_id[q_idx]
        kv_bs = graph_id[kv_idx]

        return q_bs == kv_bs

    block_mask = create_block_mask(
        causal, B=None, H=None, Q_LEN=num_tokens, KV_LEN=num_tokens
    )  # (..., ..., L, L)

    # Stack the nodes and edges
    N_jagged = torch.vstack(list_N).unsqueeze(0)  # (1, L, n_nodes)
    E_jagged = torch.vstack(list_E).unsqueeze(0)  # (1, M, n_nodes)

    return N_jagged, E_jagged


def collateGraphStatic(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
    n_max: int,
):
    """
    Data collate function to transform the outputs of the dataset into batches
    with a static shape (n_max).

    Args:
    batch, which is a list of size 2 tuples made of :
    - N: torch.Tensor, contains node of size (n_mol, natoms)
    - E: torch.Tensor, contains edge of size (n_mol, n_mol, nbonds)
    n_max: int, the fixed number of nodes to pad to.

    Returns:
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
    - nodes: torch.Tensor of shape (bs, n_max, natoms)
    - edges: torch.Tensor of shape (bs, n_max, n_max, nbonds)
    - mask: torch.Tensor of shape (bs, n_max)
    """
    list_N, list_E = zip(*batch)

    # Pad the nodes
    N_padded = torch.stack(
        [pad(N, (0, 0, 0, n_max - N.shape[0])).to(torch.float32) for N in list_N]
    )  # (bs, n_max, natoms)

    # Pad the edges
    E_padded = torch.stack(
        [
            pad(E, (0, 0, 0, n_max - E.shape[0], 0, n_max - E.shape[0])).to(
                torch.float32
            )
            for E in list_E
        ]
    )  # (bs, n_max, n_max, nbonds)

    # Make the mask
    mask = torch.stack(
        [
            torch.cat([torch.ones(N.shape[0]), torch.zeros(n_max - N.shape[0])])
            for N in list_N
        ]
    )  # (bs, n_max)

    return N_padded, E_padded, mask


def collateGraphSpec(batch: list[dict]) -> tuple:
    """
    Collate function for GraphSpecDataset.
    Handles graph padding and spectra feature collation.

    Returns:
        tuple: (nodes, edges, mask, num_peaks, types, instruments, ion_vec, form_vec, intens)
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
    )
