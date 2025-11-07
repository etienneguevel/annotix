from annotix_ml.graphtransf.test_utils import create_random_inp


def test_mask_any_tensor():
    # Choose some dimensions
    bs = 64
    n = 54
    d = 512
    de = 256

    # Create random inputs
    N, E, mask = create_random_inp(bs, n, d, de)
