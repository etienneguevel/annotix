from annotix_ml.graphtransf.data.data_utils import mask_any_tensor
from annotix_ml.graphtransf.test_utils import create_random_inp


def test_mask_any_tensor():
    # Choose some dimensions
    bs = 64
    n = 54
    d = 512
    de = 256

    # Create random inputs
    N, E, mask = create_random_inp(bs, n, d, de)

    # Do the masking with size matchings
    N_masked = N * mask.unsqueeze(-1)
    E_masked = E * mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, n, -1)

    # obtain the masked inputs with mask_any_tensor
    N_test = mask_any_tensor(N, mask)
    E_test = mask_any_tensor(E, mask)

    assert (N_masked == N_test).all()
    assert (E_masked == E_test).all()
