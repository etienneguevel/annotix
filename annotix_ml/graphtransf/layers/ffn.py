from collections import OrderedDict

import torch
import torch.nn as nn

from annotix_ml.graphtransf.data.data_utils import mask_any_tensor


class FfnNodeEdge(nn.Module):
    """
    Feed forward network of the attention layers. Nodes and Edges each have
    a MLP with one hidden layer.
    """

    def __init__(
        self,
        d: int,
        de: int,
    ):
        """
        Args:
        - d: int, the dimension of the nodes embeddings of the network.
        - de: int, the dimension of the edges embeddings of the network.
        """
        super().__init__()
        # Init the FFN for nodes
        self.d = d
        self.de = de
        self.feedforward_N = nn.Sequential(
            OrderedDict(
                [
                    ("W1", nn.Linear(d, 2 * d)),
                    ("ReLU", nn.ReLU()),
                    ("W2", nn.Linear(2 * d, d)),
                ]
            )
        )

        # Init the FFN for edges
        self.feedforward_E = nn.Sequential(
            OrderedDict(
                [
                    ("W1", nn.Linear(de, 2 * de)),
                    ("ReLU", nn.ReLU()),
                    ("W2", nn.Linear(2 * de, de)),
                ]
            )
        )

        # Make the norm layer.
        self.norm_N = nn.LayerNorm(d)
        self.norm_E = nn.LayerNorm(de)

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Compute the nodes outputs
        inter_nodes = self.feedforward_N(h) + h  # (bs, n, d)

        # Mask the nodes outputs
        inter_nodes = mask_any_tensor(inter_nodes, mask)  # (bs, n, d)
        normed_nodes = self.norm_N(inter_nodes)  # (bs, n, d)

        # Compute the edges outputs
        inter_edges = self.feedforward_E(e) + e  # (bs, n, n, de)

        # Mak the edges outputs
        inter_edges = mask_any_tensor(inter_edges, mask)  # (bs, n, n, de)
        normed_edges = self.norm_E(inter_edges)  # (bs, n, n, de)

        return normed_nodes, normed_edges, mask
