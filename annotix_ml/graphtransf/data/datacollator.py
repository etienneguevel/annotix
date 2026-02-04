import torch
from torch.nn.attention.flex_attention import create_block_mask
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence


def collateGraph(
    batch: list[tuple[torch.Tensor]],
):
    """
    Data collate function to transform the outputs of the dataset into batches
    that will be fed to the DataLoader objects during the training.
    This function is made to be used with the GraphDatasetFromSMILEs object and
    plug it to a torch DataLoader.

    Args:
    batch, which is a list of size 3 tuples made of :
    - N: torch.Tensor, contains node of size (n_mol, natoms)
    - E: torch.Tensor, contains edge of size (n_mol, n_mol, nbonds)
    - pos_emb: torch.Tensor, contains pos_emb (n_mol, k)

    Returns:
    The 3 padded stacking of the lists, as well as a mask indicating the padding
    of each element of the batch.
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

    return {"nodes": N_padded, "edges": E_padded, "node_mask": mask}


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
