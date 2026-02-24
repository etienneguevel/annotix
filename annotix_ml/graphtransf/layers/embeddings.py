import torch
import torch.nn as nn

from annotix_ml.data.data_utils import mask_any_tensor
from annotix_ml.graphtransf.layers.mlp import MLP


class EmbeddingLaplacian(nn.Module):
    """
    Embedding layer for graph-structured data.

    This layer implements separate MLPs for embedding node types, edge types,
    global features, and positional embeddings (e.g., Laplacian eigenvectors).

    Args:
        d (int): Hidden dimension for node features.
        de (int): Hidden dimension for edge features.
        dy (int): Hidden dimension for global features.
        node_features (int): Dimension of input extra node features.
        global_features (int): Dimension of input extra global features.
        natoms (int): Number of node types (e.g., atom types).
        nbonds (int): Number of edge types (e.g., bond types).
    """

    def __init__(
        self,
        d: int,
        de: int,
        dy: int,
        node_features: int,
        global_features: int,
        natoms: int,
        nbonds: int,
    ):
        """
        Initialize the EmbeddingLaplacian layer.

        Args:
            d (int): Hidden dimension for node features.
            de (int): Hidden dimension for edge features.
            dy (int): Hidden dimension for global features.
            node_features (int): Dimension of input extra node features.
            global_features (int): Dimension of input extra global features.
            natoms (int): Number of node types.
            nbonds (int): Number of edge types.
        """
        super().__init__()
        self.d = d
        self.de = de
        self.dy = dy
        self.node_features = node_features
        self.global_features = global_features
        self.natoms = natoms
        self.nbonds = nbonds
        self.EmbeddingNodes = MLP(natoms, 2 * d, d)
        self.EmbeddingEdges = MLP(nbonds, 2 * de, de)
        self.EmbeddingY = MLP(global_features, 2 * dy, dy)
        self.LaplacianProjection = MLP(node_features, 2 * d, d)

    def forward(
        self,
        N: torch.Tensor,
        E: torch.Tensor,
        global_features_t: torch.Tensor,
        node_features_t: torch.Tensor,
        mask: torch.Tensor,
    ):
        """
        Forward pass of the EmbeddingLaplacian layer.

        Args:
            N (torch.Tensor): Node features of shape (bs, n, natoms).
            E (torch.Tensor): Edge features of shape (bs, n, n, nbonds).
            global_features_t (torch.Tensor): Global extra features of shape (bs, global_features).
            node_features_t (torch.Tensor): Extra node features of shape (bs, n, node_features).
            mask (torch.Tensor): Node mask of shape (bs, n).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - h (torch.Tensor): Embedded node features of shape (bs, n, d).
                - e (torch.Tensor): Embedded edge features of shape (bs, n, n, de).
                - y (torch.Tensor): Embedded global features of shape (bs, dy).
                - mask (torch.Tensor): The input mask tensor.
        """
        # Calculate the node embedding and add the positional emb
        h = self.EmbeddingNodes(N) + self.LaplacianProjection(
            node_features_t
        )  # (bs, n, d)
        h = mask_any_tensor(h, mask)

        # Compute the edge embedding
        e = self.EmbeddingEdges(E)  # (bs, n, n, de)
        e = mask_any_tensor(e, mask)

        # Compute the y embedding
        y = self.EmbeddingY(global_features_t)  # (bs, dy)

        return h, e, y, mask


class EmbeddingLaplacianWithoutY(nn.Module):
    """
    Embedding layer for graph-structured data without explicit global features.

    Args:
        d (int): Hidden dimension for node features.
        de (int): Hidden dimension for edge features.
        node_features (int): Dimension of input extra node features.
        natoms (int): Number of node types.
        nbonds (int): Number of edge types.
    """

    def __init__(
        self,
        d: int,
        de: int,
        node_features: int,
        natoms: int,
        nbonds: int,
    ):
        """
        Initialize the EmbeddingLaplacianWithoutY layer.

        Args:
            d (int): Hidden dimension for node features.
            de (int): Hidden dimension for edge features.
            node_features (int): Dimension of input extra node features.
            natoms (int): Number of node types.
            nbonds (int): Number of edge types.
        """
        super().__init__()
        self.d = d
        self.de = de
        self.node_features = node_features
        self.natoms = natoms
        self.nbonds = nbonds
        self.EmbeddingNodes = MLP(natoms, 2 * d, d)
        self.EmbeddingEdges = MLP(nbonds, 2 * de, de)
        self.LaplacianProjection = MLP(node_features, 2 * d, d)

    def forward(
        self,
        N: torch.Tensor,
        E: torch.Tensor,
        global_features: torch.Tensor,
        node_features_extra: torch.Tensor,
        mask: torch.Tensor,
    ):
        """
        Forward pass of the EmbeddingLaplacianWithoutY layer.

        Args:
            N (torch.Tensor): Node features of shape (bs, n, natoms).
            E (torch.Tensor): Edge features of shape (bs, n, n, nbonds).
            global_features (torch.Tensor): Extra global features of shape (bs, global_features).
            node_features_extra (torch.Tensor): Extra node features of shape (bs, n, node_features).
            mask (torch.Tensor): Node mask of shape (bs, n).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - h (torch.Tensor): Embedded node features of shape (bs, n, d).
                - e (torch.Tensor): Embedded edge features of shape (bs, n, n, de).
                - mask (torch.Tensor): The input mask tensor.
        """
        bs = N.shape[0]
        n = N.shape[1]

        # Compute the edge embedding
        e = self.EmbeddingEdges(E)  # (bs, n, n, de)
        e = mask_any_tensor(e, mask)

        # Stack the features
        global_features_expanded = (
            global_features.unsqueeze(1).expand((bs, n, -1)).to(N.dtype)
        )
        features = torch.cat([node_features_extra, global_features_expanded], dim=-1)

        # Calculate the node embedding and add the positional emb
        h = self.EmbeddingNodes(N) + self.LaplacianProjection(features)  # (bs, n, d)
        h = mask_any_tensor(h, mask)

        return h, e, mask.clone()
