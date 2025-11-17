import torch
import torch.nn as nn

from annotix_ml.graphtransf.data.data_utils import mask_any_tensor


class EmbeddingLaplacian(nn.Module):
    """
    Embedding layer for graph like data. It implements two Linear layer for
    the nodes and the edges of the graph. Another layer is implemented for
    the projection of the positional embeddings (Laplacian eigenvectors).
    """

    def __init__(
        self,
        d: int,
        de: int,
        k: int,
        natoms: int,
        nbonds: int,
    ):
        """
        Args:
        - d: int, embedding dimension of the nodes.
        - de: int, embedding dimension of the edges.
        - k: int, number of eigenvectors to select.
        - natoms: int, size of the one-hot encoded nodes.
        - nbonds: int, size of the one-hot encoded edges.
        """
        super().__init__()
        self.d = d
        self.de = de
        self.k = k
        self.natoms = natoms
        self.nbonds = nbonds
        self.EmbeddingNodes = nn.Linear(natoms, d, bias=False)
        self.EmbeddingEdges = nn.Linear(nbonds, de, bias=False)
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

        Returns:
        Embedded nodes and edges. The positional embedding is added to the nodes.
        The mask is also returned unmodified.
        """
        # Calculate the node embedding and add the positional emb
        h = self.EmbeddingNodes(N) + self.LaplacianProjection(pos_emb)  # (bs, n, d)
        h = mask_any_tensor(h, mask)

        # Compute the edge embedding
        e = self.EmbeddingEdges(E)  # (bs, n, n, de)
        e = mask_any_tensor(e, mask)

        return h, e, mask


class Unembedding(nn.Module):
    """
    Unembedding layer to map the embedding of the transformer layers back to
    the one-hot encoded space of the nodes and edges.
    """

    """
    Args:
    - embedding_layer: EmbeddingLaplacian, the embedding layer of the model,
    its weights are reused to make the ones of this layer.
    """

    def __init__(self, embedding_layer):
        """
        Args:
        - embedding_layer: EmbeddingLaplacian, the embedding layer of the model,
        its weights are reused to make the ones of this layer.
        """
        super().__init__()
        self.embedding_layer = embedding_layer  # keep reference

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

        Wn = self.embedding_layer.EmbeddingNodes.weight
        We = self.embedding_layer.EmbeddingEdges.weight

        N = N @ Wn  # equivalent to Linear(d→natoms) without bias
        E = E @ We  # equivalent to Linear(de→nbonds)
        return N, E, mask
