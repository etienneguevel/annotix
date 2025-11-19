import torch

from annotix_ml.graphtransf.data.atoms_data import VALID_ELEMENTS, TYPE_EDGES
from annotix_ml.graphtransf.models.noising import NoisingModel
from annotix_ml.graphtransf.test_utils import create_random_start


def test_Q_shape():
    # Create random dist vectors
    natoms = len(VALID_ELEMENTS)
    nbonds = len(TYPE_EDGES)
    nodes_distribution = torch.randint(30, 100, (natoms,))
    edges_distribution = torch.randint(90, 250, (nbonds,))

    nodes_distribution = nodes_distribution / nodes_distribution.sum()
    edges_distribution = edges_distribution / edges_distribution.sum()

    # Init the noising layer
    model = NoisingModel(
        nodes_distribution=nodes_distribution,
        edges_distribution=edges_distribution,
        diffusion_steps=500,
    )

    # Get matrices
    Q_nodes_t, Q_edges_t = model.get_Q_t(200)
    assert Q_nodes_t.shape == (natoms, natoms)
    assert Q_edges_t.shape == (nbonds, nbonds)

    # Get bar matrices
    Q_nodes_bar_t, Q_edges_bar_t = model.get_Q_bar_t(200)
    assert Q_nodes_bar_t.shape == (natoms, natoms)
    assert Q_edges_bar_t.shape == (nbonds, nbonds)


def test_noise_forward():
    # Create random inputs
    bs = 64
    n = 30
    natoms = len(VALID_ELEMENTS)
    nbonds = len(TYPE_EDGES)
    N, E, mask = create_random_start(bs, n, nbonds, natoms)

    # Create random dist vectors
    nodes_distribution = torch.randint(30, 100, (natoms,))
    edges_distribution = torch.randint(90, 250, (nbonds,))

    nodes_distribution = nodes_distribution / nodes_distribution.sum()
    edges_distribution = edges_distribution / edges_distribution.sum()

    # Init the noising layer
    model = NoisingModel(
        nodes_distribution=nodes_distribution,
        edges_distribution=edges_distribution,
        diffusion_steps=500,
    )

    # Test the forward of the model
    N_noised, E_noised, mask_out = model(N, E, mask)
    assert N.size() == N_noised.size()
    assert E.size() == E_noised.size()
    assert mask_out is mask


if __name__ == "__main__":
    test_Q_shape()
    test_noise_forward()
