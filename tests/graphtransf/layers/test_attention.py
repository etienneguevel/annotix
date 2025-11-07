from annotix_ml.graphtransf.layers.attention import MultiHeadEdgeNode
from annotix_ml.graphtransf.test_utils import create_random_inp


def test_attention_layer():
    # Choose some dimensions
    bs = 64
    n = 54
    d = 512
    de = 256

    # Create random inputs
    N, E, mask = create_random_inp(bs, n, d, de)

    # Create the layers
    nh = 8
    attention_layer = MultiHeadEdgeNode(
        d=d,
        de=de,
        n_heads=nh,
    )

    # Test the forward method
    attn, edge_attn, mask_out = attention_layer.forward(N, E, mask)
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

    # Create random inputs
    N, E, mask = create_random_inp(bs, n, d, de)

    # Create the layers
    nh = 8
    attention_layer = MultiHeadEdgeNode(
        d=d,
        de=de,
        n_heads=nh,
    )

    # Get the 2 attention
    attn, _ = attention_layer.forward_normal(N, E, mask, attn_map_mode=True)

    assert (attn.sum(-1).round().int() == 1).all(), "attn is not normalized"


if __name__ == "__main__":
    test_attention_layer()
    test_attention_map()
