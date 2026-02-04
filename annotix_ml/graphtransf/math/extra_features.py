import torch

from annotix_ml.graphtransf.data.atoms_data import VALID_ELEMENTS, DICT_EDGES
from annotix_ml.graphtransf.data.data_utils import mask_any_tensor


def laplacian_embedding(
    edges: torch.Tensor, k: int, mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute features based on the eigenvectors of the normalized Laplacian matrix of a graph.

    Args:
        edges (torch.Tensor): Adjacency matrix of shape (bs, n, n, nbonds) or (n, n, nbonds).
        k (int): Number of eigenvectors to select.
        mask (torch.Tensor, optional): Binary mask indicating existing nodes (bs, n) or (n,).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - node_features (torch.Tensor): Laplacian node features of shape (bs, n, k+1) or (n, k+1).
                First value indicates if the node is in the largest connected component, next are the values of the node in the k eigenvectors.
            - global_features (torch.Tensor): Laplacian global features of shape (bs, k+1) or (k+1).
                First value indicates the number of connected components, next are the values of eigenvalues.
    """
    device = edges.device
    is_batched = edges.dim() == 4

    if not is_batched:
        edges = edges.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)

    # Create a mask for the diagonal values
    bs, n, *_ = edges.size()
    mask_diag = (
        2 * n * torch.eye(n, device=device).unsqueeze(0).expand(bs, -1, -1)
    )  # (bs, n, n)
    if mask is not None:
        mask_diag = mask_any_tensor(mask_diag, ~mask.int(), fill=0.0)

    # edges : (bs, n, n, nbonds)
    A = edges[..., 1:].sum(-1).float()  # Adjacency matrix (bs, n, n)

    # Apply mask to adjacency matrix if provided
    if mask is not None:
        # Mask rows: (bs, n, n) with mask (bs, n)
        A = mask_any_tensor(A, mask, fill=0.0)

    degrees = A.sum(-1)  # (bs, n)
    D_inv_sqrt = torch.diag_embed(torch.sqrt(1 / (degrees + 1e-8)))  # (bs, n, n)

    # Compute the Laplacian matrix
    eye = torch.eye(n, device=device).unsqueeze(0).expand(bs, -1, -1)
    L = eye - D_inv_sqrt @ A @ D_inv_sqrt  # (bs, n, n)
    L = L + mask_diag

    # Deal with the fact that linalg doesn't comply with mps
    if "mps" in str(device):
        L = L.to("cpu")
        if mask is not None:
            mask = mask.to("cpu")

    # Get the eigenvectors
    eigvals, eigvectors = torch.linalg.eigh(L)  # (bs, n), (bs, n, n)

    # Compute the number of connected components of the graph for each batch
    n_connected_components = (eigvals < 1e-5).sum(dim=-1)  # (bs,)

    # Look at the 1st ev
    ev1 = eigvectors[:, :, 0]  # (bs, n)
    most_common = torch.mode(ev1, dim=-1).values  # (bs,)
    mask_ev1 = ~(ev1 == most_common.unsqueeze(-1))  # (bs, n)
    not_in_ev1 = mask_ev1.unsqueeze(-1).float()  # (bs, n, 1)

    # Zero out eigenvectors for masked nodes
    if mask is not None:
        eigvectors = mask_any_tensor(eigvectors, mask, fill=0.0)
        not_in_ev1 = mask_any_tensor(not_in_ev1, mask, fill=0.0)

    # Pad the number of eigenvalues / eigenvectors in case there are not enough
    max_n_comp = n_connected_components.max().item()
    to_extend = max(0, max_n_comp + k - n)

    if to_extend > 0:
        eigvals = torch.cat(
            [
                eigvals,
                2 * torch.ones(bs, to_extend, device=device, dtype=eigvals.dtype),
            ],
            dim=-1,
        )  # (bs, n + to_extend)

        eigvectors = torch.cat(
            [
                eigvectors,
                torch.zeros(bs, n, to_extend, device=device, dtype=eigvectors.dtype),
            ],
            dim=-1,
        )  # (bs, n, n + to_extend)

    # Select the corresponding eigenvalues / eigenvectors
    eigvals_list = []
    eigvectors_list = []
    for i in range(bs):
        n_comp = n_connected_components[i].item()
        eigvals_list.append(eigvals[i, n_comp : n_comp + k])
        eigvectors_list.append(eigvectors[i, :, n_comp : n_comp + k])

    eigvals_ = torch.stack(eigvals_list, dim=0)  # (bs, k)
    eigvectors_ = torch.stack(eigvectors_list, dim=0)  # (bs, n, k)

    # Normalize the eigvals by the size of the graph
    eigvals_ = eigvals_ / n

    node_features = torch.cat(
        (
            not_in_ev1,
            eigvectors_,
        ),
        dim=-1,
    )

    global_features = torch.cat(
        (
            n_connected_components.unsqueeze(-1),
            eigvals_,
        ),
        dim=-1,
    )

    if not is_batched:
        return (
            node_features[0],
            global_features[0],
        )

    return (
        node_features.to(device),
        global_features.to(device),
    )


def batch_trace(X: torch.Tensor) -> torch.Tensor:
    """
    Compute the trace of a batch of matrices.

    Args:
        X (torch.Tensor): Batch of matrices of shape (bs, n, n).

    Returns:
        torch.Tensor: The trace of each matrix, shape (bs,).
    """
    diag = torch.diagonal(X, dim1=-2, dim2=-1)
    trace = diag.sum(dim=-1)
    return trace


def batch_diagonal(X: torch.Tensor) -> torch.Tensor:
    """
    Extract the diagonal from the last two dimensions of a tensor.

    Args:
        X (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: The diagonal elements.
    """
    return torch.diagonal(X, dim1=-2, dim2=-1)


def k3_cycle(k3_matrix):
    """
    Compute cycle counts for 3-cycles (triangles) in a graph.

    Args:
        k3_matrix: torch.Tensor of shape (..., n, n), the third power of the adjacency matrix.

    Returns:
        tuple: A tuple of two tensors:
            - First tensor: Node-level 3-cycle counts (c3 / 2), shape (..., n, 1)
            - Second tensor: Graph-level 3-cycle counts (sum(c3) / 6), shape (..., 1)
    """
    c3 = batch_diagonal(k3_matrix)
    return (c3 / 2).unsqueeze(-1).float(), (torch.sum(c3, dim=-1) / 6).unsqueeze(
        -1
    ).float()


def k4_cycle(A, d, k4_matrix):
    """
    Compute cycle counts for 4-cycles (squares) in a graph.

    Args:
        A: torch.Tensor of shape (..., n, n), the adjacency matrix.
        d: torch.Tensor of shape (..., n), the degree of each node.
        k4_matrix: torch.Tensor of shape (..., n, n), the fourth power of the adjacency matrix.

    Returns:
        tuple: A tuple of two tensors:
            - First tensor: Node-level 4-cycle counts (c4 / 2), shape (..., n, 1)
            - Second tensor: Graph-level 4-cycle counts (sum(c4) / 8), shape (..., 1)
    """
    diag_a4 = batch_diagonal(k4_matrix)
    c4 = diag_a4 - d * (d - 1) - (A @ d.unsqueeze(-1)).sum(dim=-1)
    return (c4 / 2).unsqueeze(-1).float(), (torch.sum(c4, dim=-1) / 8).unsqueeze(
        -1
    ).float()


def k5_cycle(A, d, k3_matrix, k5_matrix):
    """
    Compute cycle counts for 5-cycles (pentagons) in a graph.

    Args:
        A: torch.Tensor of shape (..., n, n), the adjacency matrix.
        d: torch.Tensor of shape (..., n), the degree of each node.
        k3_matrix: torch.Tensor of shape (..., n, n), the third power of the adjacency matrix (used to extract triangle counts).
        k5_matrix: torch.Tensor of shape (..., n, n), the fifth power of the adjacency matrix.

    Returns:
        tuple: A tuple of two tensors:
            - First tensor: Node-level 5-cycle counts (c5 / 2), shape (..., n, 1)
            - Second tensor: Graph-level 5-cycle counts (sum(c5) / 10), shape (..., 1)
    """
    diag_a5 = batch_diagonal(k5_matrix)
    triangles = batch_diagonal(k3_matrix)
    c5 = (
        diag_a5
        - 2 * triangles * d
        - (A @ triangles.unsqueeze(-1)).sum(dim=-1)
        + triangles
    )
    return (c5 / 2).unsqueeze(-1).float(), (c5.sum(dim=-1) / 10).unsqueeze(-1).float()


def k6_cycle(A, k2_matrix, k3_matrix, k4_matrix, k6_matrix):
    """
    Compute cycle counts for 6-cycles (hexagons) in a graph.

    Args:
        A: torch.Tensor of shape (..., n, n), the adjacency matrix.
        k2_matrix: torch.Tensor of shape (..., n, n), the second power of the adjacency matrix.
        k3_matrix: torch.Tensor of shape (..., n, n), the third power of the adjacency matrix.
        k4_matrix: torch.Tensor of shape (..., n, n), the fourth power of the adjacency matrix.
        k6_matrix: torch.Tensor of shape (..., n, n), the sixth power of the adjacency matrix.

    Returns:
        tuple: A tuple of two tensors:
            - First tensor: None (node-level 6-cycle counts are not computed)
            - Second tensor: Graph-level 6-cycle counts (c6 / 12), shape (..., 1)
    """
    term_1_t = batch_trace(k6_matrix)
    term_2_t = batch_trace(k3_matrix**2)
    term3_t = torch.sum(A * k2_matrix.pow(2), dim=[-2, -1])
    d_t4 = batch_diagonal(k2_matrix)
    a_4_t = batch_diagonal(k4_matrix)
    term_4_t = (d_t4 * a_4_t).sum(dim=-1)
    term_5_t = batch_trace(k4_matrix)
    term_6_t = batch_trace(k3_matrix)
    term_7_t = batch_diagonal(k2_matrix).pow(3).sum(-1)
    term8_t = torch.sum(k3_matrix, dim=[-2, -1])
    term9_t = batch_diagonal(k2_matrix).pow(2).sum(-1)
    term10_t = batch_trace(k2_matrix)

    c6_t = (
        term_1_t
        - 3 * term_2_t
        + 9 * term3_t
        - 6 * term_4_t
        + 6 * term_5_t
        - 4 * term_6_t
        + 4 * term_7_t
        + 3 * term8_t
        - 12 * term9_t
        + 4 * term10_t
    )
    return None, (c6_t / 12).unsqueeze(-1).float()


def node_cycle(edges: torch.Tensor, mask: torch.Tensor | None = None):
    """
    Compute cycle counts of sizes 3, 4, 5, and 6 for a graph.

    Args:
        edges (torch.Tensor): Edge tensor of shape (bs, n, n, nbonds) or (n, n, nbonds).
        mask (torch.Tensor, optional): Binary mask indicating existing nodes (bs, n) or (n,).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - kcyclesx (torch.Tensor): Node-level cycle counts for 3, 4, and 5-cycles.
            - kcyclesy (torch.Tensor): Graph-level cycle counts for 3, 4, 5, and 6-cycles.
    """
    is_batched = edges.dim() == 4

    if not is_batched:
        edges = edges.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)

    # edges : (bs, n, n, nbonds)
    A = edges[..., 1:].sum(-1).float()  # Adjacency matrix (bs, n, n)

    # Apply mask to adjacency matrix if provided
    if mask is not None:
        # Mask rows: (bs, n, n) with mask (bs, n)
        A = mask_any_tensor(A, mask, fill=0.0)

    d = A.sum(-1)  # (bs, n)

    # Compute the cycles of size 3
    k2_matrix = A @ A
    k3_matrix = k2_matrix @ A
    k3x, k3y = k3_cycle(k3_matrix)

    # Compute the cycles of size 4
    k4_matrix = k3_matrix @ A
    k4x, k4y = k4_cycle(A, d, k4_matrix)

    # Compute the cyles of size 5
    k5_matrix = k4_matrix @ A
    k5x, k5y = k5_cycle(A, d, k3_matrix, k5_matrix)

    # Compute the cycles of size 6
    k6_matrix = k5_matrix @ A
    _, k6y = k6_cycle(A, k2_matrix, k3_matrix, k4_matrix, k6_matrix)

    kcyclesx = torch.cat([k3x, k4x, k5x], dim=-1) / 10  # (bs, n, 3)
    kcyclesy = torch.cat([k3y, k4y, k5y, k6y], dim=-1) / 10  # (bs, 4)

    kcyclesx = kcyclesx.clamp(min=0.0, max=1.0)
    kcyclesy = kcyclesy.clamp(min=0.0, max=1.0)

    # Zero out node-level features for masked nodes
    if mask is not None:
        kcyclesx = mask_any_tensor(kcyclesx, mask, fill=0.0)

    if not is_batched:
        return kcyclesx[0], kcyclesy[0]

    return kcyclesx, kcyclesy  # (bs, n, 3), (bs, 4)


def valency(edges: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute the valency of each node in the graph.

    Args:
        edges (torch.Tensor): Edge tensor of shape (bs, n, n, nedges).
        mask (torch.Tensor): Binary mask for nodes of shape (bs, n).

    Returns:
        torch.Tensor: Node valency features of shape (bs, n, 1).
    """
    # edges: bs, n, n, nedges
    # mask: bs, n

    bond_valence = torch.tensor(
        list(DICT_EDGES.values()),
        device=edges.device,
        dtype=edges.dtype,
    )  # nedges

    valence = (edges @ bond_valence).sum(-1).unsqueeze(-1)  # bs, n, 1
    valence = mask_any_tensor(valence, mask)

    return valence


def charge(
    nodes: torch.Tensor,
    edges: torch.Tensor,
    mask: torch.Tensor,
    valid_elements: list[str],
) -> torch.Tensor:
    """
    Compute the formal charge of each node in the graph.

    Args:
        nodes (torch.Tensor): Node features of shape (bs, n, natoms).
        edges (torch.Tensor): Edge features of shape (bs, n, n, nedges).
        mask (torch.Tensor): Binary mask for nodes of shape (bs, n).
        valid_elements (list[str]): List of valid atomic element symbols.

    Returns:
        torch.Tensor: Node charge features of shape (bs, n, 1).
    """
    # nodes: bs, n, natoms
    # edges: bs, n, n, nedges
    # mask: bs, n
    cov_mat = torch.tensor(
        [getattr(VALID_ELEMENTS.get(at), "covalence") for at in valid_elements],
        device=nodes.device,
        dtype=nodes.dtype,
    )  # natoms

    valence_th = (nodes @ cov_mat).unsqueeze(-1)  # bs, n, 1

    # Compute the actual covalence
    valence = valency(edges, mask)  # bs, n, 1

    # Compute the diff
    ch = valence - valence_th  # bs, n, 1
    ch = mask_any_tensor(ch, mask)

    return ch


def weight(
    nodes: torch.Tensor, valid_elements: list[str], max_weight: float | None = None
) -> torch.Tensor:
    """
    Compute the total molecular weight of the graph.

    Args:
        nodes (torch.Tensor): Node features of shape (bs, n, natoms).
        valid_elements (list[str]): List of valid atomic element symbols.
        max_weight (float, optional): Maximum weight for normalization.

    Returns:
        torch.Tensor: Molecular weight feature of shape (bs, 1).
    """
    # nodes: bs, n, natoms
    # mask: bs, n

    weight_tensor = torch.tensor(
        [getattr(VALID_ELEMENTS[at], "weight") for at in valid_elements],
        device=nodes.device,
        dtype=nodes.dtype,
    )  # natoms
    weight_mols = (nodes @ weight_tensor).sum(-1).unsqueeze(-1)  # bs, 1

    if max_weight:
        weight_mols = weight_mols / max_weight

    return weight_mols
