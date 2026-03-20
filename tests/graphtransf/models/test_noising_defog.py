import pytest
import torch
from annotix_ml.graphtransf.models.noising_defog import NoisingModelDefog
from annotix_ml.graphtransf.test_utils import create_random_start


class TestNoisingModelDefog:
    @pytest.fixture(params=["uniform", "marginal"])
    def model(self, request):
        natoms = 5
        nbonds = 4
        nodes_dist = torch.ones(natoms) / natoms
        edges_dist = torch.ones(nbonds) / nbonds
        return NoisingModelDefog(
            nodes_distribution=nodes_dist,
            edges_distribution=edges_dist,
            diffusion_steps=10,
            transition_type=request.param,
        )

    # ------------------------------------------------------------------
    # compute_noised_graph
    # ------------------------------------------------------------------

    def test_compute_noised_graph_output_sums_to_one(self, model):
        bs, n = 3, 8
        N, E, mask = create_random_start(bs, n, model.nbonds, model.natoms)
        t = 5

        N_noised, E_noised = model.compute_noised_graph(N.float(), E.float(), t, mask)

        # Output should be one-hot (sum = 1 over last dim) for valid positions
        node_mask_bool = mask.bool()
        assert torch.allclose(
            N_noised[node_mask_bool].sum(-1),
            torch.ones(node_mask_bool.sum()),
            atol=1e-5,
        )

    def test_compute_noised_graph_at_t0_is_limit_distribution(self):
        """At t=0, the noised graph should be sampled from the limit distribution."""
        torch.manual_seed(0)
        natoms = 5
        nbonds = 4
        nodes_dist = torch.ones(natoms) / natoms
        edges_dist = torch.ones(nbonds) / nbonds
        model = NoisingModelDefog(
            nodes_distribution=nodes_dist,
            edges_distribution=edges_dist,
            diffusion_steps=100,
            transition_type="uniform",
        )
        # Large batch to get stable statistics
        bs, n = 500, 4
        N, E, mask = create_random_start(bs, n, nbonds, natoms)

        N_noised, E_noised = model.compute_noised_graph(N.float(), E.float(), 0, mask)

        # At t=0 the probability passed to the sampler is purely limit_X (uniform)
        # so each class should appear with roughly equal frequency
        counts = N_noised[mask.bool()].sum(0)  # (natoms,)
        expected = counts.sum() / natoms
        assert (counts - expected).abs().max() < expected * 0.3, (
            f"Node class counts not roughly uniform at t=0: {counts}"
        )

    def test_compute_noised_graph_at_tT_equals_input(self, model):
        """At t=T, valid positions in the noised graph should equal the input."""
        bs, n = 4, 6
        N, E, mask = create_random_start(bs, n, model.nbonds, model.natoms)
        T = model.T

        N_noised, E_noised = model.compute_noised_graph(N.float(), E.float(), T, mask)

        # Only check valid (non-masked) node positions
        node_mask_bool = mask.bool()  # (bs, n)
        assert torch.equal(N_noised[node_mask_bool], N.float()[node_mask_bool])

        # Only check valid edge positions (both endpoints must be valid)
        edge_mask_bool = (mask.unsqueeze(-1) * mask.unsqueeze(-2)).bool()  # (bs, n, n)
        # Edges are symmetric so also check upper triangular only
        upper = torch.triu(edge_mask_bool.float(), diagonal=1).bool()
        assert torch.equal(E_noised[upper], E.float()[upper])

    def test_compute_noised_graph_unbatched(self, model):
        """Unbatched input (no batch dim) should work without error."""
        n = 6
        N = torch.zeros(n, model.natoms)
        N[:, 0] = 1.0
        E = torch.zeros(n, n, model.nbonds)
        E[:, :, 0] = 1.0

        N_noised, E_noised = model.compute_noised_graph(N, E, 5)

        assert N_noised.shape == (n, model.natoms)
        assert E_noised.shape == (n, n, model.nbonds)

    # ------------------------------------------------------------------
    # get_posterior
    # ------------------------------------------------------------------

    def test_get_posterior_shapes(self, model):
        bs, n = 2, 8
        N, E, mask = create_random_start(bs, n, model.nbonds, model.natoms)
        t = 5

        pN, pE = model.get_posterior(N.float(), E.float(), t)

        assert pN.shape == (bs, n, model.natoms, model.natoms)
        assert pE.shape == (bs, n, n, model.nbonds, model.nbonds)

    def test_get_posterior_rows_sum_to_one(self, model):
        """Each row (over last dim) of the posterior should sum to 1."""
        bs, n = 2, 8
        N, E, mask = create_random_start(bs, n, model.nbonds, model.natoms)

        pN, pE = model.get_posterior(N.float(), E.float(), t=5)

        assert torch.allclose(pN.sum(-1), torch.ones_like(pN.sum(-1)), atol=1e-5)
        assert torch.allclose(pE.sum(-1), torch.ones_like(pE.sum(-1)), atol=1e-5)

    def test_get_posterior_at_t1_is_limit_distribution(self, model):
        """At t=1 (s=0), posterior should equal π_∞ for all rows."""
        bs, n = 2, 6
        N, E, mask = create_random_start(bs, n, model.nbonds, model.natoms)

        pN, pE = model.get_posterior(N.float(), E.float(), t=1)

        # s = (1-1)/T = 0 → posterior = π_∞ regardless of x_1
        limit_X = model.limit_X
        limit_E = model.limit_E

        # Each row of pN should equal limit_X
        for k in range(model.natoms):
            row = pN[0, 0, k, :]  # p(x_s | x_1=k)
            assert torch.allclose(row, limit_X, atol=1e-5), (
                f"Row {k} at t=1 != limit_X: {row} vs {limit_X}"
            )

        for k in range(model.nbonds):
            row = pE[0, 0, 0, k, :]
            assert torch.allclose(row, limit_E, atol=1e-5), (
                f"Row {k} at t=1 != limit_E: {row} vs {limit_E}"
            )

    def test_get_posterior_at_tT_approaches_identity(self, model):
        """At t=T (s=(T-1)/T), the diagonal should dominate."""
        bs, n = 2, 6
        N, E, _ = create_random_start(bs, n, model.nbonds, model.natoms)
        T = model.T

        pN, pE = model.get_posterior(N.float(), E.float(), t=T)

        s = (T - 1) / T
        I_N = torch.eye(model.natoms)

        # The posterior should be close to identity (modulo the small (1-s)*limit_X term)
        limit_X = model.limit_X
        expected = s * I_N + (1 - s) * limit_X.unsqueeze(0)
        assert torch.allclose(pN[0, 0], expected, atol=1e-5)

    def test_get_posterior_nonnegative(self, model):
        bs, n = 2, 8
        N, E, _ = create_random_start(bs, n, model.nbonds, model.natoms)

        pN, pE = model.get_posterior(N.float(), E.float(), t=5)

        assert (pN >= 0).all()
        assert (pE >= 0).all()

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def test_forward_returns_correct_shapes(self, model):
        bs, n = 4, 10
        N, E, mask = create_random_start(bs, n, model.nbonds, model.natoms)

        N_noised, E_noised, sampled_t = model(N.float(), E.float(), mask)

        assert N_noised.shape == N.shape
        assert E_noised.shape == E.shape
        assert sampled_t.shape == (bs, 1)

    def test_forward_sampled_t_in_range(self, model):
        bs, n = 4, 10
        N, E, mask = create_random_start(bs, n, model.nbonds, model.natoms)

        _, _, sampled_t = model(N.float(), E.float(), mask)

        assert (sampled_t >= 1).all()
        assert (sampled_t < model.T).all()

    # ------------------------------------------------------------------
    # Absorbing transition type
    # ------------------------------------------------------------------

    def test_absorbing_increments_natoms_nbonds(self):
        natoms = 5
        nbonds = 4
        model = NoisingModelDefog(
            nodes_distribution=torch.ones(natoms) / natoms,
            edges_distribution=torch.ones(nbonds) / nbonds,
            diffusion_steps=10,
            transition_type="absorbing",
        )
        assert model.natoms == natoms + 1
        assert model.nbonds == nbonds + 1
        assert model.limit_X[-1] == 1.0
        assert model.limit_E[-1] == 1.0
