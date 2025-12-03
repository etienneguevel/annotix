from functools import partial

import torch

from annotix_ml.graphtransf.data.atoms_data import VALID_ELEMENTS, TYPE_EDGES
from annotix_ml.graphtransf.layers import (
    EmbeddingLaplacian,
    FfnNodeEdge,
    MultiHeadEdgeNodeWithY,
    Unembedding,
    MLPNodeEdge,
)
from annotix_ml.graphtransf.math.extra_features import laplacian_embedding, node_cycle
from annotix_ml.graphtransf.models.gnn import GnnNodeEdges
from annotix_ml.graphtransf.test_utils import create_random_start


def test_model_creation():
    d = 512
    de = 256
    dy = 128
    n_heads = 8
    k = 20
    n_layers = 4
    natoms = len(VALID_ELEMENTS)
    nbonds = len(TYPE_EDGES)

    # Init the model with Unembedding
    model = GnnNodeEdges(
        d=d,
        de=de,
        dy=dy,
        n_heads=n_heads,
        node_features=k,
        global_features=k + 1,
        n_layers=n_layers,
        natoms=natoms,
        nbonds=nbonds,
        last_layer="unembedding",
    )
    embedding_layer = model.layers.pop(0)
    unembedding_layer = model.layers.pop(-1)

    assert type(embedding_layer) is EmbeddingLaplacian
    assert (embedding_layer.d == d) & (embedding_layer.node_features == k)

    assert type(unembedding_layer) is Unembedding
    assert unembedding_layer.embedding_layer is embedding_layer

    assert len(model.layers) == n_layers
    for layer in model.layers:
        if type(attn := layer.attnEdgeNode) is MultiHeadEdgeNodeWithY:
            assert (attn.d == d) & (attn.de == de) & (attn.n_heads == n_heads)

        elif type(ffn := layer.ffnEdgeNode) is FfnNodeEdge:
            assert (ffn.d == d) & (ffn.de == de)

        else:
            raise TypeError(f"{type(layer)} is not comprehended yet.")

    # Init the model with MLP
    model = GnnNodeEdges(
        d=d,
        de=de,
        dy=dy,
        n_heads=n_heads,
        node_features=k,
        global_features=k + 1,
        n_layers=n_layers,
        natoms=natoms,
        nbonds=nbonds,
        last_layer="mlp",
    )
    assert isinstance(model.layers[-1], MLPNodeEdge)


def test_model_forward():
    d = 512
    de = 256
    dy = 128
    n_heads = 8
    k = 8
    n_layers = 4
    natoms = len(VALID_ELEMENTS)
    nbonds = len(TYPE_EDGES)

    # Create random input tensors
    bs = 64
    n = 54
    N, E, mask = create_random_start(bs, n, nbonds, natoms)
    y = torch.randn((bs, k + 4))

    # get the extra features
    extra_features_functions = []

    f = partial(laplacian_embedding, k=k)
    extra_features_functions.append(f)
    extra_features_functions.append(node_cycle)

    node_features, global_features = list(
        zip(
            *[
                extra_feature(edges=E, mask=mask)
                for extra_feature in extra_features_functions
            ]
        )
    )

    pos_emb = torch.cat(node_features, dim=-1)  # (bs, n, n_features)
    y = torch.cat(global_features, dim=-1)  # (bs, n_global_features)

    # Init the model
    model = GnnNodeEdges(
        d=d,
        de=de,
        dy=dy,
        n_heads=n_heads,
        node_features=k + 3,
        global_features=k + 4,
        n_layers=n_layers,
        natoms=natoms,
        nbonds=nbonds,
    )

    # Test the forward function
    h, e, mask = model.forward(N, E, pos_emb, y, mask)
