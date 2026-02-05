import torch
from annotix_ml.graphtransf.models.gnn import GnnNodeEdges
from annotix_ml.graphtransf.test_utils import create_random_start


def test_gnn_residual_connection():
    # Model parameters
    d = 16
    de = 8
    dy = 16
    n_heads = 2
    node_features = 4
    global_features = 4
    n_layers = 2
    natoms = 5
    nbonds = 3

    # Initialize model
    model = GnnNodeEdges(
        d=d,
        de=de,
        dy=dy,
        n_heads=n_heads,
        node_features=node_features,
        global_features=global_features,
        n_layers=n_layers,
        natoms=natoms,
        nbonds=nbonds,
        last_layer="mlp",
    )

    # Set all parameters to zero to make the network function as a zero-mapping
    # This ensures that output = 0 + input
    for p in model.parameters():
        p.data.fill_(0.0)

    model.eval()

    # Create dummy inputs
    bs = 2
    n = 10

    # Random inputs
    N, E, mask = create_random_start(bs, n, nbonds, natoms)
    N = N.float()
    E = E.float()

    pos_emb = torch.randn(bs, n, node_features)
    y = torch.randn(bs, global_features)  # Not used in residual but needed for forward

    # Forward pass
    batch = {
        "nodes": N,
        "edges": E,
        "node_features": pos_emb,
        "global_features": y,
        "mask": mask,
    }
    batch = model(batch)
    h_out, e_out = batch["nodes"], batch["edges"]

    # Expected output for h is N (since network output is 0)
    # But we need to consider that the network output might be masked?
    # The residual is added to the network output.
    # If network output is 0, then h_out should be N.

    # Check nodes
    # We expect h_out to be approximately N
    # Note: The model might apply masking to the output.
    # Let's check if the unmasked parts match.

    # Verify shape
    assert h_out.shape == N.shape
    assert e_out.shape == E.shape

    # Verify values
    # For nodes: h_out should be N
    # We use a small tolerance because of float precision, though with 0 weights it should be exact.
    torch.testing.assert_close(h_out, N)

    # For edges: e_out should be E * diag_mask
    # Construct diag_mask
    # diag_mask = torch.eye(n)
    # diag_mask = ~diag_mask.bool()
    # diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, nbonds)

    # expected_E = E * diag_mask.float()

    # The model also symmetrizes edges: e = 1/2 * (e + e.T)
    # The residual is added: e = (network_out + E_in) * diag_mask
    # Then symmetrized: e = 1/2 * (e + e.T)

    # If network_out is 0:
    # e_before_sym = E_in * diag_mask
    # e_final = 1/2 * (e_before_sym + e_before_sym.transpose(1, 2))

    expected_E_sym = 0.5 * (E + E.transpose(1, 2))

    torch.testing.assert_close(e_out, expected_E_sym)


if __name__ == "__main__":
    test_gnn_residual_connection()
