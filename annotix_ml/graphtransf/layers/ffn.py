import torch
import torch.nn as nn

from annotix_ml.graphtransf.data.data_utils import mask_any_tensor


class Ffn(nn.Module):
    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.feedforward = nn.ModuleList(
            [
                nn.Linear(d, 2 * d),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(2 * d, d),
                nn.Dropout(dropout),
            ]
        )

    def forward(self, x):
        for layer in self.feedforward:
            x = layer(x)
        return x


class FfnNodeEdge(nn.Module):
    """
    Feed-forward network for node and edge features.

    This layer applies separate MLPs with one hidden layer to node and edge features,
    followed by layer normalization and residual connections.

    Args:
        d (int): Hidden dimension for node features.
        de (int): Hidden dimension for edge features.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
    """

    def __init__(
        self,
        d: int,
        de: int,
        dropout: float = 0.1,
    ):
        """
        Initialize the FfnNodeEdge layer.

        Args:
            d (int): Hidden dimension for node features.
            de (int): Hidden dimension for edge features.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super().__init__()
        # Init the FFN for nodes
        self.d = d
        self.de = de
        self.feedforward_N = Ffn(d, dropout)

        # Init the FFN for edges
        self.feedforward_E = Ffn(de, dropout)

        # Make the norm layer.
        self.norm_N = nn.LayerNorm(d)
        self.norm_E = nn.LayerNorm(de)

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the FfnNodeEdge layer.

        Args:
            h (torch.Tensor): Node features of shape (bs, n, d).
            e (torch.Tensor): Edge features of shape (bs, n, n, de).
            mask (torch.Tensor): Node mask of shape (bs, n).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - h (torch.Tensor): Updated node features of shape (bs, n, d).
                - e (torch.Tensor): Updated edge features of shape (bs, n, n, de).
                - mask (torch.Tensor): The input mask tensor.
        """
        # Compute the nodes outputs, make residual connection
        inter_nodes = self.feedforward_N(h) + h  # (bs, n, d)

        # Mask the nodes outputs
        inter_nodes = mask_any_tensor(inter_nodes, mask)  # (bs, n, d)
        normed_nodes = self.norm_N(inter_nodes)  # (bs, n, d)

        # Compute the edges outputs, make residual connection
        inter_edges = self.feedforward_E(e) + e  # (bs, n, n, de)

        # Mak the edges outputs
        inter_edges = mask_any_tensor(inter_edges, mask)  # (bs, n, n, de)
        normed_edges = self.norm_E(inter_edges)  # (bs, n, n, de)

        return normed_nodes, normed_edges, mask
