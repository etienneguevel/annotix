import torch
import torch.nn as nn

from annotix_ml.graphtransf.data.atoms_data import VALID_ELEMENTS, TYPE_EDGES
from annotix_ml.graphtransf.layers import (
    AttentionLayer,
    AttentionLayerWithoutY,
    EmbeddingLaplacian,
    EmbeddingLaplacianWithoutY,
    MLPNodeEdge,
    MLPNodeEdgeWithoutY,
    Unembedding,
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
        last_layer (str, optional): Type of the last layer ("mlp" or "unembedding"). Defaults to "mlp".
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
        last_layer: str = "mlp",
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
            last_layer (str, optional): Type of the last layer ("mlp" or "unembedding"). Defaults to "mlp".
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
        if last_layer == "mlp":
            layers.append(MLPNodeEdge(d, de, natoms, nbonds))

        elif last_layer == "unembedding":
            layers.append(Unembedding(layers[0]))

        else:
            raise ValueError(f"Unknown last layer: {last_layer}")

        # Build the unembedding layer
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass of the GNN model.

        Args:
            batch (dict[str, torch.Tensor]): A dictionary containing:
                - "nodes" (torch.Tensor): Node features of shape (bs, n, natoms).
                - "edges" (torch.Tensor): Edge features of shape (bs, n, n, nedges).
                - "node_features" (torch.Tensor): Extra node features of shape (bs, n, node_features).
                - "global_features" (torch.Tensor): Extra global features of shape (bs, global_features).
                - "mask" (torch.Tensor): Mask tensor of shape (bs, n).

        Returns:
            dict[str, torch.Tensor]: The input batch dictionary with updated keys:
                - "nodes" (torch.Tensor): Updated node features of shape (bs, n, d).
                - "edges" (torch.Tensor): Updated edge features of shape (bs, n, n, de).
        """
        # h -> nodes (bs, n, d)
        # e -> edges (bs, n, n, de)
        # y -> global_features (bs, dy)
        h = batch["nodes"]
        e = batch["edges"]
        mask = batch["mask"]
        node_features = batch["node_features"]
        global_features = batch["global_features"]

        # Capture inputs for residual connection
        N_in = h.clone()
        E_in = e.clone()

        for layer in self.layers:
            if isinstance(layer, EmbeddingLaplacian):
                h, e, y, mask = layer(h, e, global_features, node_features, mask)
                # Symmetrize edges at the beginning
                e = 1 / 2 * (e + e.transpose(1, 2))  # (bs, n, n, de)

            else:
                h, e, y, mask = layer(h, e, y, mask)

            # Symmetrize the edges matrices
            e = 1 / 2 * (e + e.transpose(1, 2))

        # Add residual connections (skip connection from input to output)
        h = h + N_in
        e = e + E_in

        # Re-symmetrize after adding residual
        e = 1 / 2 * (e + e.transpose(1, 2))

        # Return the batch with updated node and edge features
        batch["nodes"] = h
        batch["edges"] = e
        batch["mask"] = mask

        return batch


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
        last_layer (str, optional): Type of the last layer ("mlp" or "unembedding"). Defaults to "mlp".
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
        last_layer: str = "mlp",
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
            last_layer (str, optional): Type of the last layer ("mlp" or "unembedding"). Defaults to "mlp".
        """
        super().__init__()
        self.d = d
        self.de = de
        self.n_layers = n_layers

        # Make the Embedding layer
        layers: list[nn.Module] = [
            EmbeddingLaplacianWithoutY(
                d, de, node_features + global_features, natoms, nbonds
            )
        ]

        # Build the attention layers
        for _ in range(n_layers):
            layers.append(AttentionLayerWithoutY(d, de, n_heads))

        # Build the last layer
        if last_layer == "mlp":
            layers.append(MLPNodeEdgeWithoutY(d, de, natoms, nbonds))

        elif last_layer == "unembedding":
            layers.append(Unembedding(layers[0]))

        else:
            raise ValueError(f"Unknown last layer: {last_layer}")

        # Build the unembedding layer
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass of the GNN model without global feature updates.

        Args:
            batch (dict[str, torch.Tensor]): A dictionary containing:
                - "nodes" (torch.Tensor): Node features of shape (bs, n, natoms).
                - "edges" (torch.Tensor): Edge features of shape (bs, n, n, nedges).
                - "node_features" (torch.Tensor): Extra node features of shape (bs, n, node_features).
                - "global_features" (torch.Tensor): Extra global features of shape (bs, global_features).
                - "mask" (torch.Tensor): Mask tensor of shape (bs, n).

        Returns:
            dict[str, torch.Tensor]: The input batch dictionary with updated keys:
                - "nodes" (torch.Tensor): Updated node features of shape (bs, n, d).
                - "edges" (torch.Tensor): Updated edge features of shape (bs, n, n, de).
        """
        # h -> nodes (bs, n, d)
        # e -> edges (bs, n, n, de)
        # y -> global_features (bs, dy)
        # node_features -> node_features (bs, n, node_features)
        h = batch["nodes"]
        e = batch["edges"]
        mask = batch["mask"]
        node_features = batch["node_features"]
        global_features = batch["global_features"]

        bs = h.shape[0]
        n = h.shape[1]

        # Capture inputs for residual connection
        N_in = h.clone()
        E_in = e.clone()
        global_features = global_features.unsqueeze(1).expand((bs, n, -1))
        features = torch.cat([node_features, global_features], dim=-1)

        for layer in self.layers:
            if isinstance(layer, EmbeddingLaplacianWithoutY):
                h, e, mask = layer(h, e, features, mask)
                # Symmetrize edges at the beginning
                e = 1 / 2 * (e + e.transpose(1, 2))  # (bs, n, n, de)

            else:
                h, e, mask = layer(h, e, mask)

            # Symmetrize the edges matrices
            e = 1 / 2 * (e + e.transpose(1, 2))

        # Add residual connections (skip connection from input to output)
        h = h + N_in
        e = e + E_in

        # Re-symmetrize after adding residual
        e = 1 / 2 * (e + e.transpose(1, 2))

        # Return the batch with updated node and edge features
        batch["nodes"] = h
        batch["edges"] = e
        batch["mask"] = mask

        return batch


# Make different size of the model
def gnnNodeEdgesBase() -> GnnNodeEdges:
    model = GnnNodeEdges(
        d=256,
        de=64,
        dy=256,
        n_heads=8,
        node_features=13,
        global_features=14,
        n_layers=5,
        natoms=len(VALID_ELEMENTS),
        nbonds=len(TYPE_EDGES),
    )
    return model
