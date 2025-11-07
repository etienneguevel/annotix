import torch

from annotix_ml.graphtransf.data.atoms_data import VALID_ELEMENTS, TYPE_EDGES
from annotix_ml.graphtransf.layers import (
    EmbeddingLaplacian,
    FfnNodeEdge,
    MultiHeadEdgeNode,
    Unembedding,
)
from annotix_ml.graphtransf.math.positional_emb import laplacian_embedding
from annotix_ml.graphtransf.models.gnn import GnnNodeEdges
from annotix_ml.graphtransf.test_utils import create_random_start


def test_model_creation():
    d = 512
    de = 256
    n_heads = 8
    k = 20
    n_layers = 4
    natoms = len(VALID_ELEMENTS)
    nbonds = len(TYPE_EDGES)

    # Init the model
    model = GnnNodeEdges(
        d=d,
        de=de,
        n_heads=n_heads,
        k=k,
        n_layers=n_layers,
        natoms=natoms,
        nbonds=nbonds,
    )
    embedding_layer = model.layers.pop(0)
    unembedding_layer = model.layers.pop(-1)

    assert type(embedding_layer) is EmbeddingLaplacian
    assert (embedding_layer.d == d) & (embedding_layer.k == k)

    assert type(unembedding_layer) is Unembedding
    assert unembedding_layer.embedding_layer is embedding_layer

    assert len(model.layers) == n_layers
    for layer in model.layers:
        if type(attn := layer.attnEdgeNode) is MultiHeadEdgeNode:
            assert (attn.d == d) & (attn.de == de) & (attn.n_heads == n_heads)

        elif type(ffn := layer.ffnEdgeNode) is FfnNodeEdge:
            assert (ffn.d == d) & (ffn.de == de)

        else:
            raise TypeError(f"{type(layer)} is not comprehended yet.")


def test_model_forward():
    d = 512
    de = 256
    n_heads = 8
    k = 8
    n_layers = 4
    natoms = len(VALID_ELEMENTS)
    nbonds = len(TYPE_EDGES)

    # Init the model
    model = GnnNodeEdges(
        d=d,
        de=de,
        n_heads=n_heads,
        k=k,
        n_layers=n_layers,
        natoms=natoms,
        nbonds=nbonds,
    )

    # Create random input tensors
    bs = 64
    n = 54
    N, E, mask = create_random_start(bs, n, nbonds, natoms)

    # Create the embeddings
    eigv = []
    for edges, m in zip(E, mask):
        n_mol = int(m.sum(-1))
        adj = edges[:n_mol, :n_mol, :]
        eigvectors = laplacian_embedding(adj, k)
        eigv.append(eigvectors)

    eigv = torch.nested.nested_tensor(eigv, layout=torch.jagged)
    pos_emb = eigv.to_padded_tensor(padding=0, output_size=(bs, n, k))

    # Test the forward function
    h, e, mask = model.forward(N, E, pos_emb, mask)


if __name__ == "__main__":
    test_model_creation()
    test_model_forward()
