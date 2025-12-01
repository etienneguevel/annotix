import pytest
import torch
from annotix_ml.graphtransf.models.noising import NoisingModel
from annotix_ml.graphtransf.test_utils import create_random_start


class TestNoisingModel:
    @pytest.fixture
    def model(self):
        natoms = 5
        nbonds = 4
        nodes_dist = torch.ones(natoms) / natoms
        edges_dist = torch.ones(nbonds) / nbonds
        return NoisingModel(
            nodes_distribution=nodes_dist,
            edges_distribution=edges_dist,
            diffusion_steps=10,
            noise_schedule_type="cosine",
        )

    def test_get_posterior_shapes_and_values(self, model):
        bs = 2
        n_nodes = 20
        natoms = model.natoms
        nbonds = model.nbonds

        # Create random one-hot encoded inputs using test utility
        N, E, mask = create_random_start(bs, n_nodes, nbonds, natoms)

        t = 5  # Arbitrary step > 0

        pN_posterior, pE_posterior = model.get_posterior(N, E, t)

        # Check shapes
        # Expected N posterior: (bs, n, natoms, natoms)
        assert pN_posterior.shape == (bs, n_nodes, natoms, natoms)

        # Expected E posterior: (bs, n, n, nbonds, nbonds)
        assert pE_posterior.shape == (bs, n_nodes, n_nodes, nbonds, nbonds)

        # Check values are probabilities (sum to 1 along last dim)
        # We use allclose because of floating point precision
        # Only check non-masked nodes (where mask == 1)

        # For non-masked nodes, check that probabilities sum to 1
        for b in range(bs):
            for n in range(n_nodes):
                if mask[b, n] > 0:  # Only check non-masked nodes
                    assert torch.allclose(
                        pN_posterior[b, n].sum(dim=-1),
                        torch.ones(natoms),
                        atol=1e-5,
                    )

        # For edges, check non-masked edges (both nodes must be non-masked)
        mask_edge = mask.unsqueeze(-1) * mask.unsqueeze(-2)  # (bs, n, n)
        for b in range(bs):
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if mask_edge[b, i, j] > 0:  # Only check non-masked edges
                        assert torch.allclose(
                            pE_posterior[b, i, j].sum(dim=-1),
                            torch.ones(nbonds),
                            atol=1e-5,
                        )

        # Check values are in [0, 1]
        assert (pN_posterior >= -1e-6).all() and (pN_posterior <= 1.0 + 1e-6).all()
        assert (pE_posterior >= -1e-6).all() and (pE_posterior <= 1.0 + 1e-6).all()
