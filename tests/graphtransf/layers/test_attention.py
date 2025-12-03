import torch
from annotix_ml.graphtransf.layers.attention import (
    MultiHeadEdgeNodeWithY,
    AttentionLayer,
)
from annotix_ml.graphtransf.test_utils import create_random_inp


def test_attention_layer():
    # Choose some dimensions
    bs = 64
    n = 54
    d = 512
    de = 256
    dy = 128

    # Create random inputs
    N, E, mask = create_random_inp(bs, n, d, de)
    y = torch.randn((bs, dy))

    # Create the layers
    nh = 8
    attention_layer = MultiHeadEdgeNodeWithY(
        d=d,
        de=de,
        dy=dy,
        n_heads=nh,
    )

    # Test the forward method
    attn, edge_attn, mask_out = attention_layer.forward(N, E, y, mask)
    assert attn.size() == N.size(), f"The size of the attn vector is {attn.size()}"
    assert edge_attn.size() == E.size(), (
        f"The size of the edge attn vector is {edge_attn.size()}"
    )
    assert mask is mask_out


def test_attention_map():
    # Choose some dimensions
    bs = 64
    n = 54
    d = 512
    de = 256
    dy = 128

    # Create random inputs
    N, E, mask = create_random_inp(bs, n, d, de)
    y = torch.randn((bs, dy))

    # Create the layers
    nh = 8
    attention_layer = MultiHeadEdgeNodeWithY(
        d=d,
        de=de,
        dy=dy,
        n_heads=nh,
    )

    # Get the 2 attention
    _, attn = attention_layer.forward_normal(N, E, y, mask, attn_map_mode=True)

    assert (attn.sum(-1).round().int() == 1).all(), "attn is not normalized"


def test_full_attention_layer():
    # Choose some dimensions
    bs = 64
    n = 54
    d = 512
    de = 256
    dy = 128

    # Create random inputs
    N, E, mask = create_random_inp(bs, n, d, de)
    y = torch.randn((bs, dy))

    # Create the full attention layer (attention + ffn)
    nh = 8
    full_attention_layer = AttentionLayer(
        d=d,
        de=de,
        dy=dy,
        n_heads=nh,
    )

    # Test the forward method
    h_out, e_out, y_out, mask_out = full_attention_layer.forward(N, E, y, mask)
    assert h_out.size() == N.size(), f"The size of the output nodes is {h_out.size()}"
    assert e_out.size() == E.size(), f"The size of the output edges is {e_out.size()}"
    assert y_out.size() == y.size(), f"The size of the output y is {y_out.size()}"
    assert mask is mask_out


def test_masked_attention():
    # Choose some dimensions
    bs = 2
    n = 20
    d = 32
    de = 16
    dy = 8
    nh = 4

    # Create random inputs
    N, E, mask = create_random_inp(bs, n, d, de)
    y = torch.randn((bs, dy))

    # Manually set the mask to have some padded nodes
    # Let's say the last 2 nodes of the first graph are padded
    mask[0, -2:] = 0
    # And the last 4 nodes of the second graph are padded
    mask[1, -4:] = 0

    # Create the layer
    attention_layer = MultiHeadEdgeNodeWithY(
        d=d,
        de=de,
        dy=dy,
        n_heads=nh,
    )

    # Get the attention map
    attn, node_attn = attention_layer.forward_normal(N, E, y, mask, attn_map_mode=True)

    # Check that padded nodes receive 0 attention
    # node_attn shape: (bs, nh, n, n)
    # We check attention *to* padded nodes (last dim)

    # For graph 0
    attn_to_padded_0 = node_attn[0, :, :, -2:]
    assert (attn_to_padded_0 == 0).all(), "Graph 0: Padded nodes received attention!"

    # For graph 1
    attn_to_padded_1 = node_attn[1, :, :, -4:]
    assert (attn_to_padded_1 == 0).all(), "Graph 1: Padded nodes received attention!"
