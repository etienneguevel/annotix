import torch


def laplacian_embedding(edges: torch.Tensor, k: int) -> torch.Tensor:
    """
    Fonction to compute the eigenvectors of the normalized Laplacian matrix of
    the edges of a graph.
    Returns the k non-0 eigenvectors.
    Args:
        - edges: torch.Tensor, matrix of adjacency of the graph (n, n, nbonds)
        - k: int, number of eigenvectors to use (should be inf to n-1)

    Returns:
    tensor of dim (n, k) corresponding to the k smallest non-0 eigenvectors.
    """
    # edges (n, n, nbonds)

    device = edges.device
    n, *_ = edges.size()

    # edges : (n, n, nbonds)
    A = edges[..., 1:].sum(-1).float()  # Adjacency matrix (n, n)
    D = torch.diag(torch.sqrt(1 / A.sum(-1)))  # Degree matrix ** -0.5 (n, n)

    # Compute the Laplacian matrix
    L = torch.eye(n) - D @ A @ D  # Laplacian matrix (n, n)

    # Deal with the fact that linalg doesn't comply with mps
    if str(device) == "mps":
        L = L.to("cpu")

    # Get the eigenvectors
    eigvals, eigvectors = torch.linalg.eigh(L)  # (n), (n, n)

    eigvals = eigvals.to(device)
    eigvectors = eigvectors.to(device)

    return eigvectors[:, 1 : k + 1]  # return the k smallest eigvectors
