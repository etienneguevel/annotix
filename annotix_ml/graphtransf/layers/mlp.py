import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, d: int, d_hidden: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPNodeEdge(nn.Module):
    def __init__(self, d: int, d_e: int, natoms: int, nedges: int):
        super().__init__()
        self.MLPN = MLP(d, 2 * d, natoms)
        self.MLPE = MLP(d_e, 2 * d_e, nedges)

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.MLPN(h), self.MLPE(e), y, mask
