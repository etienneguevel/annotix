import torch
import torch.nn as nn

from annotix_ml.graphtransf.data.atoms_data import VALID_ELEMENTS, TYPE_EDGES
from annotix_ml.graphtransf.layers import AttentionLayer, EmbeddingLaplacian

class GnnNodeEdges(nn.Module):
    def __init__(
        self,
        d: int,
        de: int,
        n_heads: int,
        k: int,
        n_layers: int,
        natoms: int,
        nbonds: int,
    ):
        super().__init__()
        self.d = d
        self.de = de
        self.n_layers = n_layers

        layers = [
            EmbeddingLaplacian(d, de, k, natoms, nbonds)
        ]

        for _ in range(n_layers):
            layers.append(AttentionLayer(d, de, n_heads))
        
        self.layers = nn.ModuleList(layers)


    def forward(
        self,
        N: torch.Tensor,
        E: torch.Tensor,
        pos_emb: torch.Tensor,
        mask: torch.Tensor,
    ):
        # Do the forward pass
        for i, layer in enumerate(self.layers):
            if i == 0:
                h, e, mask = layer(N, E, pos_emb, mask)
            
            else:
                h, e, mask = layer(h, e, mask)

        return h, e, mask

def gnnNodeEdgesBase() -> GnnNodeEdges:
    model = GnnNodeEdges(
        d=256,
        de=64,
        n_heads=8,
        k=5,
        n_layers=5,
        natoms=len(VALID_ELEMENTS),
        nbonds=len(TYPE_EDGES),
    )
    return model
