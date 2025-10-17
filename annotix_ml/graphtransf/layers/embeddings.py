import torch
import torch.nn as nn
import torch.nn.functional as F

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
        e = self.EmbeddingEdges(E) # (bs, n, n, de)
        e = e * edge_mask

        return h, e, mask
    

class Unembedding(nn.Module):
    def __init__(
        self,
        embedding_layer: EmbeddingLaplacian,
    ):
        super().__init__()
        self.UnembedNodes = embedding_layer.EmbeddingNodes.weight.T
        self.UnembedEdges = embedding_layer.EmbeddingEdges.weight.T

    def forward(
        self,
        N: torch.Tensor,
        E: torch.Tensor,
        mask: torch.Tensor,
    ):
        """
        Args:
            - N: torch.Tensor, node matrix (bs, n, d)
            - E: torch.Tensor, adjacency matrix (bs, n, n, de)
            - mask: torch.Tensor, boolean mask (bs, n)
        """
        # Compute the logits
        N = F.linear(N, self.UnembedNodes) # (bs, n, natoms)
        E = F.linear(E, self.UnembedEdges) # (bs, n, n, nbonds)

        # TODO : find a way to mask the atoms not related to the element of the 
        # batch for the cross entropy
        
        return N, E, mask