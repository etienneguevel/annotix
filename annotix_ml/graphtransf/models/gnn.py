import torch
import torch.nn as nn

from annotix_ml.graphtransf.layers import (
    AttentionLayer,
    AttentionLayerWithoutY,
    EmbeddingLaplacian,
    EmbeddingLaplacianWithoutY,
    MLPNodeEdge,
    MLPNodeEdgeWithoutY,
)


class GnnNodeEdges(nn.Module):
    """
    Implementation of the GNN model as described in https://arxiv.org/abs/2012.09699.

    This model processes graph-structured data (nodes and edges) along with global features,
    using a series of attention layers to update the node, edge, and global representations.

    Args:
        d (int): Hidden dimension for node features.
        de (int): Hidden dimension for edge features.
        dy (int): Hidden dimension for global features.
        n_heads (int): Number of attention heads in the attention layers.
        node_features (int): Dimension of input extra node features.
        global_features (int): Dimension of input extra global features.
        n_layers (int): Number of attention layers.
        natoms (int): Number of output node types (e.g., atom types).
        nbonds (int): Number of output edge types (e.g., bond types).
        y_update (bool, optional): Whether to update global features in the attention layers. Defaults to True.
    """

    def __init__(
        self,
        d: int,
        de: int,
        dy: int,
        n_heads: int,
        node_features: int,
        global_features: int,
        n_layers: int,
        natoms: int,
        nbonds: int,
        y_update: bool = True,
    ):
        """
        Initialize the GnnNodeEdges model.

        Args:
            d (int): Hidden dimension for node features.
            de (int): Hidden dimension for edge features.
            dy (int): Hidden dimension for global features.
            n_heads (int): Number of attention heads.
            node_features (int): Dimension of input extra node features.
            global_features (int): Dimension of input extra global features.
            n_layers (int): Number of attention layers.
            natoms (int): Number of output node types.
            nbonds (int): Number of output edge types.
            y_update (bool, optional): Whether to update global features. Defaults to True.
        """
        super().__init__()
        self.d = d
        self.de = de
        self.dy = dy
        self.n_layers = n_layers

        # Make the Embedding layer
        layers: list[nn.Module] = [
            EmbeddingLaplacian(
                d, de, dy, node_features, global_features, natoms, nbonds
            )
        ]

        # Build the attention layers
        for _ in range(n_layers):
            layers.append(AttentionLayer(d, de, dy, n_heads, y_update))

        # Build the last layer
        layers.append(MLPNodeEdge(d, de, natoms, nbonds))

        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        y: torch.Tensor,
        node_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the GNN model.

        Args:
            h (torch.Tensor): Node features of shape (bs, n, natoms).
            e (torch.Tensor): Edge features of shape (bs, n, n, nedges).
            y (torch.Tensor): Global features of shape (bs, dy).
            node_features (torch.Tensor): Extra node features of shape (bs, n, node_features).
            mask (torch.Tensor): Mask tensor of shape (bs, n).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - h (torch.Tensor): Updated node features of shape (bs, n, d).
                - e (torch.Tensor): Updated edge features of shape (bs, n, n, de).
                - y (torch.Tensor): Updated global features of shape (bs, dy).
                - mask (torch.Tensor): Mask tensor (unchanged).
        """
        # y is already computed/provided in this version
        for layer in self.layers:
            if isinstance(layer, EmbeddingLaplacian):
                h, e, y, mask = layer(h, e, y, node_features, mask)

            else:
                h, e, y, mask = layer(h, e, y, mask)

        return h, e, y, mask


class GnnNodeEdgesWithoutY(nn.Module):
    """
    Implementation of the GNN model as described in https://arxiv.org/abs/2012.09699,
    without explicit global feature updates in the attention layers.

    This model is similar to `GnnNodeEdges` but does not maintain or update a separate
    global feature vector `y` within its attention layers. Instead, global features
    are typically concatenated with node features at the beginning.

    Args:
        d (int): Hidden dimension for node features.
        de (int): Hidden dimension for edge features.
        n_heads (int): Number of attention heads.
        node_features (int): Dimension of input extra node features.
        global_features (int): Dimension of input extra global features.
        n_layers (int): Number of attention layers.
        natoms (int): Number of output node types.
        nbonds (int): Number of output edge types.
    """

    def __init__(
        self,
        d: int,
        de: int,
        n_heads: int,
        node_features: int,
        global_features: int,
        n_layers: int,
        natoms: int,
        nbonds: int,
    ):
        """
        Initialize the GnnNodeEdgesWithoutY model.

        Args:
            d (int): Hidden dimension for node features.
            de (int): Hidden dimension for edge features.
            n_heads (int): Number of attention heads.
            node_features (int): Dimension of input extra node features.
            global_features (int): Dimension of input extra global features.
            n_layers (int): Number of attention layers.
            natoms (int): Number of output node types.
            nbonds (int): Number of output edge types.
        """
        super().__init__()
        self.d = d
        self.de = de
        self.n_layers = n_layers

        # Make the Embedding layer
        self.embedding_layer = EmbeddingLaplacianWithoutY(
            d, de, node_features + global_features, natoms, nbonds
        )

        # Build the attention layers
        layers = [AttentionLayerWithoutY(d, de, n_heads) for _ in range(n_layers)]

        # Build the last layer
        layers.append(MLPNodeEdgeWithoutY(d, de, natoms, nbonds))

        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        e: torch.Tensor,
        mask: torch.Tensor,
        global_features: torch.Tensor,
        node_features: torch.Tensor,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the GNN model without global feature updates.

        Args:
            h (torch.Tensor): Node features of shape (bs, n, natoms).
            e (torch.Tensor): Edge features of shape (bs, n, n, nedges).
            global_features (torch.Tensor): Extra global features of shape (bs, global_features).
            node_features (torch.Tensor): Extra node features of shape (bs, n, node_features).
            mask (torch.Tensor): Mask tensor of shape (bs, n).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - h (torch.Tensor): Updated node features of shape (bs, n, d).
                - e (torch.Tensor): Updated edge features of shape (bs, n, n, de).
                - mask (torch.Tensor): Mask tensor (unchanged).
        """
        if self.embedding_layer is not None:
            h, e, mask = self.embedding_layer(
                h, e, global_features, node_features, mask
            )

        global_features_ = global_features.clone()
        node_features_ = node_features.clone()

        for layer in self.layers:
            h, e, mask = layer(h, e, mask)

        return e, mask, global_features_, node_features_, h
