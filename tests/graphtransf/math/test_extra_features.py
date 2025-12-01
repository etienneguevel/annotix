import torch

from annotix_ml.graphtransf.data.atoms_data import VALID_ELEMENTS, TYPE_EDGES
from annotix_ml.graphtransf.math.extra_features import laplacian_embedding, node_cycle
from annotix_ml.graphtransf.test_utils import create_random_start


def test_laplacian_emb():
    n = 40
    bs = 64
    natoms = len(VALID_ELEMENTS)
    nbonds = len(TYPE_EDGES)
    k = 5

    N, E, mask = create_random_start(bs, n, nbonds, natoms)

    # Test single graph without mask
    for edges, m in zip(E, mask):
        n_mol = int(m.sum(-1))
        adj = edges[:n_mol, :n_mol, :]
        eigvectors, eigvals = laplacian_embedding(adj, k)

        assert eigvals.shape == (k,), f"expected {(k,)} got {eigvals.shape}"
        assert eigvectors.shape == (n_mol, k), (
            f"expected {(n_mol, k)} got {eigvectors.shape}"
        )

    # Test single graph with mask
    for edges, m in zip(E, mask):
        eigvectors, eigvals = laplacian_embedding(edges, k, mask=m)

        assert eigvals.shape == (k,), f"expected {(k,)} got {eigvals.shape}"
        assert eigvectors.shape == (n, k), f"expected {(n, k)} got {eigvectors.shape}"

        # Check that masked nodes have zero eigenvectors
        n_mol = int(m.sum(-1))
        if n_mol < n:
            assert torch.allclose(
                eigvectors[n_mol:], torch.zeros_like(eigvectors[n_mol:])
            ), "Masked nodes should have zero eigenvectors"

    # Test batched graphs without mask
    eigvecs_batch, eigvals_batch = laplacian_embedding(E, k)
    assert eigvals_batch.shape == (bs, k), (
        f"expected {(bs, k)} got {eigvals_batch.shape}"
    )
    assert eigvecs_batch.shape == (bs, n, k), (
        f"expected {(bs, n, k)} got {eigvecs_batch.shape}"
    )

    # Test batched graphs with mask
    eigvecs_batch_m, eigvals_batch_m = laplacian_embedding(E, k, mask=mask)
    assert eigvals_batch_m.shape == (bs, k), (
        f"expected {(bs, k)} got {eigvals_batch_m.shape}"
    )
    assert eigvecs_batch_m.shape == (bs, n, k), (
        f"expected {(bs, n, k)} got {eigvecs_batch_m.shape}"
    )


def test_laplacian_emb_known_matrix():
    """
    Construct a simple 2-node graph with adjacency A = [[0,1],[1,0]]
    For this graph the normalized Laplacian has eigenvectors [1, 1]/sqrt(2)
    (eigenvalue 0) and [1, -1]/sqrt(2) (non-zero). The function returns the
    non-zero eigenvectors, so we check that the returned vector is equal to
    [1, -1]/sqrt(2) up to sign.
    """

    # Build edges tensor with two bond types (index 0 ignored by laplacian_embedding)
    n = 2
    nbonds = 2
    edges = torch.zeros((n, n, nbonds), dtype=torch.float32)

    # Put adjacency in bond index 1 so A = edges[..., 1]
    A = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    edges[:, :, 1] = A

    # Request the single non-zero eigenvector (single graph without mask)
    eigvecs, eigvals = laplacian_embedding(edges, k=1)

    assert eigvecs.shape == (n, 1)

    vec = eigvecs[:, 0]
    # normalize to unit norm (should already be normalized by eigh)
    vec = vec / vec.norm()

    expected = torch.tensor([1.0, -1.0], dtype=torch.float32)
    expected = expected / expected.norm()

    # eigenvectors are defined up to sign: check absolute dot product ~ 1
    dot = torch.abs(torch.dot(vec, expected))
    assert torch.isclose(dot, torch.tensor(1.0), atol=1e-5), (
        f"expected eigenvector parallel to {expected.tolist()}, got {vec.tolist()} (|dot|={dot.item()})"
    )

    # Test with batched version (no mask)
    edges_batch = edges.unsqueeze(0)
    eigvecs_b, eigvals_b = laplacian_embedding(edges_batch, k=1)

    assert eigvecs_b.shape == (1, n, 1)
    assert torch.allclose(eigvecs, eigvecs_b[0], atol=1e-5), (
        "Single and batched results should match"
    )

    # Test with mask (all nodes active)
    mask_full = torch.ones(n, dtype=torch.float32)
    eigvecs_m, eigvals_m = laplacian_embedding(edges, k=1, mask=mask_full)
    assert torch.allclose(eigvecs, eigvecs_m, atol=1e-5), (
        "Results with full mask should match no mask"
    )

    # Test with partial mask (only first node active)
    mask_partial = torch.tensor([1.0, 0.0], dtype=torch.float32)
    eigvecs_p, eigvals_p = laplacian_embedding(edges, k=1, mask=mask_partial)
    assert eigvecs_p[1, 0].item() == 0.0, "Masked node should have zero eigenvector"


def test_node_cycle():
    """Test node_cycle function with and without mask."""
    n = 20
    bs = 8
    natoms = len(VALID_ELEMENTS)
    nbonds = len(TYPE_EDGES)

    N, E, mask = create_random_start(bs, n, nbonds, natoms)

    # Test single graph without mask
    for edges, m in zip(E, mask):
        n_mol = int(m.sum(-1))
        adj = edges[:n_mol, :n_mol, :]
        kcyclesx, kcyclesy = node_cycle(adj)

        assert kcyclesx.shape == (n_mol, 3), (
            f"expected {(n_mol, 3)} got {kcyclesx.shape}"
        )
        assert kcyclesy.shape == (4,), f"expected {(4,)} got {kcyclesy.shape}"

    # Test single graph with mask
    for edges, m in zip(E, mask):
        kcyclesx, kcyclesy = node_cycle(edges, mask=m)

        assert kcyclesx.shape == (n, 3), f"expected {(n, 3)} got {kcyclesx.shape}"
        assert kcyclesy.shape == (4,), f"expected {(4,)} got {kcyclesy.shape}"

        # Check that masked nodes have zero cycle counts
        n_mol = int(m.sum(-1))
        if n_mol < n:
            assert torch.allclose(
                kcyclesx[n_mol:], torch.zeros_like(kcyclesx[n_mol:])
            ), "Masked nodes should have zero cycle counts"

    # Test batched graphs with mask
    kcyclesx_batch_m, kcyclesy_batch_m = node_cycle(E, mask=mask)
    assert kcyclesx_batch_m.shape == (bs, n, 3), (
        f"expected {(bs, n, 3)} got {kcyclesx_batch_m.shape}"
    )
    assert kcyclesy_batch_m.shape == (bs, 4), (
        f"expected {(bs, 4)} got {kcyclesy_batch_m.shape}"
    )
