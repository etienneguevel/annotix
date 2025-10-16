import torch
import torch.nn as nn

class EmbeddingLaplacian(nn.Module):

    def __init__(
        self,
        d: int,
        de: int,
        k: int,
        natoms: int,
        nbonds: int,
    ):
        super().__init__()
        
        self.d = d
        self.de = de
        self.k = k
        self.EmbeddingNodes = nn.Linear(natoms, d)
        self.EmbeddingEdges = nn.Linear(nbonds, de)
        self.LaplacianProjection = nn.Linear(k, d)

    def forward(
        self,
        N: torch.Tensor,
        E: torch.Tensor,
        pos_emb: torch.Tensor,
        mask: torch.Tensor,
    ):
        """
        Args:
            - N: torch.Tensor, node matrix (bs, n, natoms)
            - E: torch.Tensor, adjacency matrix (bs, n, n, nbonds)
            - pos_emb: torch.Tensor, Laplacian eigenvectors (bs, n, k)
            - mask: torch.Tensor, boolean mask (bs, n)
        """
        # Prepare the masks
        node_mask = mask.unsqueeze(-1) # (bs, n, 1)
        edge_mask = node_mask.unsqueeze(-1) # (bs, n, 1, 1)

        # Calculate the node embedding and add the positional emb
        h = self.EmbeddingNodes(N) + self.LaplacianProjection(pos_emb) # (bs, n, d)
        h = h * node_mask

        # Compute the edge embedding
        e = self.EmbeddingEdges(E) # (bs, n, n, d)
        e = e * edge_mask

        return h, e, mask