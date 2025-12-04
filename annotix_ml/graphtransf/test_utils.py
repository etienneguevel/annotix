import torch
import torch.nn.functional as F


def create_random_inp(bs, n, d, de):
    # Create the fake matrices
    N = torch.randn((bs, n, d))  # (bs, n, d)
    E = torch.randn((bs, n, n, de))  # (bs, n, n, de)
    mask = torch.randint(low=16, high=n, size=(bs,))
    mask = torch.concatenate(
        [torch.cat([torch.ones(m), torch.zeros(n - m)]) for m in mask]
    ).view((bs, n))

    return N, E, mask


def create_random_start(bs, n, nbonds, natoms):
    # Select random values of nodes and edges
    nodes = torch.randint(high=natoms, size=(bs, n))  # (bs, n)
    edges = torch.randint(high=nbonds, size=(bs, n, n))  # (bs, n, n)

    # Make the one-hot
    N = F.one_hot(nodes)  # (bs, n, natoms)
    E = F.one_hot(edges)  # (bs, n, n, nbonds)

    # Make a mask
    mask = torch.randint(low=1, high=n, size=(bs,))
    mask = torch.stack(
        [torch.cat([torch.ones(m), torch.zeros(n - m)]) for m in mask]
    )  # (bs, n)

    # Mask N and E
    N = N * mask.unsqueeze(-1)  # (bs, n, natoms)
    E = E * mask.unsqueeze(-1).unsqueeze(-1)  # (bs, n, n, nbonds)

    return N, E, mask
