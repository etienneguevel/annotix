import torch
from unittest.mock import MagicMock, patch
from annotix_ml.graphtransf.arch.digress_meta_arch import DigressMetaArch


@patch("annotix_ml.graphtransf.arch.digress_meta_arch.auto_model_split")
@patch("annotix_ml.graphtransf.arch.digress_meta_arch.ScheduleGPipe")
@patch("annotix_ml.graphtransf.arch.digress_meta_arch.get_global_rank")
def test_setup_distributed_microbatch_slicing(mock_get_rank, mock_schedule, mock_split):
    # Setup mocks
    mock_get_rank.return_value = 0
    mock_split.return_value = (MagicMock(), MagicMock())

    # Initialize DigressMetaArch with dummy values
    # We mock everything that's not needed for _setup_distributed
    d, de, dy = 16, 16, 16
    n_heads, n_layers = 1, 1
    diffusion_steps = 10
    loss_ratio = 1.0
    device = torch.device("cpu")
    valid_elements = ["C", "O"]

    # Mock noiser
    noiser = MagicMock()
    noiser.T = diffusion_steps
    noiser.move_to = MagicMock()

    with patch(
        "annotix_ml.graphtransf.arch.digress_meta_arch.NoisingModel",
        return_value=noiser,
    ):
        model = DigressMetaArch(
            d,
            de,
            dy,
            n_heads,
            n_layers,
            "uniform",
            diffusion_steps,
            loss_ratio,
            device,
            valid_elements,
            extra_features=[],
        )

    # Create dummy example batch
    bs = 8
    num_microbatches = 4
    mb_size = bs // num_microbatches

    n_atoms = len(valid_elements)
    n_edges = 5  # arbitrary
    max_nodes = 10

    N = torch.randn(bs, max_nodes, n_atoms)
    E = torch.randn(bs, max_nodes, max_nodes, n_edges)
    mask = torch.ones(bs, max_nodes)
    example_batch = (N, E, mask)

    # Call _setup_distributed
    model._setup_distributed("pipeline", num_microbatches, example_batch)

    # Verify that iterative_model_split was called with micro-batch sized inputs
    args, kwargs = mock_split.call_args
    example_input = args[1]

    # example_input = (N_mb, E_mb, y, pos_emb, mask_mb)
    N_mb, E_mb, y, pos_emb, mask_mb = example_input

    assert N_mb.shape[0] == mb_size
    assert E_mb.shape[0] == mb_size
    assert mask_mb.shape[0] == mb_size
    assert y.shape[0] == mb_size
    assert pos_emb.shape[0] == mb_size

    # Verify ScheduleGPipe was called for both train and eval schedules
    assert mock_schedule.call_count == 2

    # First call: train_schedule (with loss_fn)
    # Second call: eval_schedule (without loss_fn)
    calls = mock_schedule.call_args_list
    assert "loss_fn" in calls[0][1]
    assert "loss_fn" not in calls[1][1]


if __name__ == "__main__":
    test_setup_distributed_microbatch_slicing()
