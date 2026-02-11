import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multilayer Perceptron (MLP) with one hidden layer and SiLU activation.

    Args:
        d (int): Input dimension.
        d_hidden (int): Hidden layer dimension.
        d_out (int): Output dimension.
    """

    def __init__(self, d: int, d_hidden: int, d_out: int):
        """
        Initialize the MLP.

        Args:
            d (int): Input dimension.
            d_hidden (int): Hidden layer dimension.
            d_out (int): Output dimension.
        """
        super().__init__()
        self.d = d
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.net = nn.Sequential(
            nn.Linear(d, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.net(x)


class MLPNodeEdge(nn.Module):
    """
    MLP layer for joint node and edge processing.

    Args:
        d (int): Hidden dimension for node features.
        de (int): Hidden dimension for edge features.
        natoms (int): Number of node types.
        nedges (int): Number of edge types.
    """

    def __init__(self, d: int, de: int, natoms: int, nedges: int):
        """
        Initialize the MLPNodeEdge layer.

        Args:
            d (int): Hidden dimension for node features.
            de (int): Hidden dimension for edge features.
            natoms (int): Number of node types.
            nedges (int): Number of edge types.
        """
        super().__init__()
        self.MLPN = MLP(d, 2 * d, natoms)
        self.MLPE = MLP(de, 2 * de, nedges)

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the MLPNodeEdge layer.

        Args:
            h (torch.Tensor): Node features of shape (bs, n, d).
            e (torch.Tensor): Edge features of shape (bs, n, n, de).
            y (torch.Tensor): Global features of shape (bs, dy).
            mask (torch.Tensor): Node mask of shape (bs, n).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - h (torch.Tensor): Processed node features.
                - e (torch.Tensor): Processed edge features.
                - y (torch.Tensor): The input global features.
                - mask (torch.Tensor): The input mask tensor.
        """
        h, e = self.MLPN(h), self.MLPE(e)
        e = 0.5 * (e + e.transpose(1, 2))
        return h, e, y, mask


class MLPNodeEdgeWithoutY(nn.Module):
    """
    MLP layer for joint node and edge processing without global features.

    Args:
        d (int): Hidden dimension for node features.
        de (int): Hidden dimension for edge features.
        natoms (int): Number of node types.
        nedges (int): Number of edge types.
    """

    def __init__(self, d: int, de: int, natoms: int, nedges: int):
        """
        Initialize the MLPNodeEdgeWithoutY layer.

        Args:
            d (int): Hidden dimension for node features.
            de (int): Hidden dimension for edge features.
            natoms (int): Number of node types.
            nedges (int): Number of edge types.
        """
        super().__init__()
        self.MLPN = MLP(d, 2 * d, natoms)
        self.MLPE = MLP(de, 2 * de, nedges)

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the MLPNodeEdgeWithoutY layer.

        Args:
            h (torch.Tensor): Node features of shape (bs, n, d).
            e (torch.Tensor): Edge features of shape (bs, n, n, de).
            mask (torch.Tensor): Node mask of shape (bs, n).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - h (torch.Tensor): Processed node features.
                - e (torch.Tensor): Processed edge features.
                - mask (torch.Tensor): The input mask tensor.
        """
        h, e = self.MLPN(h), self.MLPE(e)
        e = 0.5 * (e + e.transpose(1, 2))
        return h, e, mask
