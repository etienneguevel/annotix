import torch
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence

# TODO : implement tests
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
    list_N, list_E, list_pos_emb = zip(*batch)

    # Look at the number of atoms in each graph
    num_atoms = [N.shape[0] for N in list_N]
    n_batch = max(num_atoms)

    # Pad the nodes
    N_padded = pad_sequence(list_N).transpose(0, 1).to(torch.float32) # (bs, n_batch, natoms)

    # Pad the embeddings
    pos_emb_padded = pad_sequence(list_pos_emb).transpose(0, 1) # (bs, n_batch, k)

    # Pad the edges
    E_padded = torch.cat(
        [
            pad(E, (0, 0, 0, n_batch - n, 0, n_batch - n)).unsqueeze(0)
            for n, E in zip(num_atoms, list_E)
        ],
        dim=0
    ).to(torch.float32) # (bs, n_batch, n_batch, k)

    # Make the mask
    mask = torch.stack(
        [
            torch.cat([torch.ones(m), torch.zeros(n_batch-m)])
            for m in num_atoms
        ]
    ) # (bs, n_batch)

    return N_padded, E_padded, pos_emb_padded, mask
