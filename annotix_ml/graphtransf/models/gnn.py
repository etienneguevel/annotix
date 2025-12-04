import torch
import torch.nn as nn

from annotix_ml.graphtransf.data.atoms_data import VALID_ELEMENTS, TYPE_EDGES
from annotix_ml.graphtransf.layers import (
    AttentionLayer,
    EmbeddingLaplacian,
    MLPNodeEdge,
    Unembedding,
)


class GnnNodeEdges(nn.Module):
    """
    Implementation of the GNN model as described in https://arxiv.org/abs/2012.09699.
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
        last_layer: str = "mlp",
    ):
        super().__init__()
        self.d = d
        self.de = de
        self.dy = dy
        self.n_layers = n_layers

        # Make the Embedding layer
        layers = [
            EmbeddingLaplacian(
                d, de, dy, node_features, global_features, natoms, nbonds
            )
        ]

        # Build the attention layers
        for _ in range(n_layers):
            layers.append(AttentionLayer(d, de, dy, n_heads))

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
        N: torch.Tensor,
        E: torch.Tensor,
        pos_emb: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
    ):
        # h -> nodes (bs, n, d)
        # e -> edges (bs, n, n, de)
        # y -> global_features (bs, dy)

        for i, layer in enumerate(self.layers):
            if i == 0:
                h, e, y, mask = layer(N, E, y, pos_emb, mask)

            else:
                h, e, y, mask = layer(h, e, y, mask)

            # Symmetrize the edges matrices
            e = 1 / 2 * (e + e.transpose(1, 2))

        return h, e, mask


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
