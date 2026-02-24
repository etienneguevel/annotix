import torch

from annotix_ml.data.atoms_data import VALID_ELEMENTS, TYPE_EDGES
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
        node_feat, global_feat = laplacian_embedding(adj, k)
        # node_feat: (n_mol, k+1), global_feat: (k+1,)

        assert global_feat.shape == (k + 1,), (
            f"expected {(k + 1,)} got {global_feat.shape}"
        )
        assert node_feat.shape == (n_mol, k + 1), (
            f"expected {(n_mol, k + 1)} got {node_feat.shape}"
        )

        eigvals = global_feat[..., 1:]
        eigvectors = node_feat[..., 1:]

        assert eigvals.shape == (k,), f"expected {(k,)} got {eigvals.shape}"
        assert eigvectors.shape == (n_mol, k), (
            f"expected {(n_mol, k)} got {eigvectors.shape}"
        )

    # Test single graph with mask
    for edges, m in zip(E, mask):
        node_feat, global_feat = laplacian_embedding(edges, k, mask=m)

        assert global_feat.shape == (k + 1,), (
            f"expected {(k + 1,)} got {global_feat.shape}"
        )
        assert node_feat.shape == (n, k + 1), (
            f"expected {(n, k + 1)} got {node_feat.shape}"
        )

        eigvals = global_feat[..., 1:]
        eigvectors = node_feat[..., 1:]

        assert eigvals.shape == (k,), f"expected {(k,)} got {eigvals.shape}"
        assert eigvectors.shape == (n, k), f"expected {(n, k)} got {eigvectors.shape}"

        # Check that masked nodes have zero eigenvectors
        n_mol = int(m.sum(-1))
        if n_mol < n:
            assert torch.allclose(
                eigvectors[n_mol:], torch.zeros_like(eigvectors[n_mol:])
            ), "Masked nodes should have zero eigenvectors"

    # Test batched graphs without mask
    node_feat_batch, global_feat_batch = laplacian_embedding(E, k)
    assert global_feat_batch.shape == (bs, k + 1), (
        f"expected {(bs, k + 1)} got {global_feat_batch.shape}"
    )
    assert node_feat_batch.shape == (bs, n, k + 1), (
        f"expected {(bs, n, k + 1)} got {node_feat_batch.shape}"
    )

    # Test batched graphs with mask
    node_feat_batch_m, global_feat_batch_m = laplacian_embedding(E, k, mask=mask)
    assert global_feat_batch_m.shape == (bs, k + 1), (
        f"expected {(bs, k + 1)} got {global_feat_batch_m.shape}"
    )
    assert node_feat_batch_m.shape == (bs, n, k + 1), (
        f"expected {(bs, n, k + 1)} got {node_feat_batch_m.shape}"
    )


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
