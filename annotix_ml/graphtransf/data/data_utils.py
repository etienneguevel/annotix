import torch


def mask_any_tensor(
    t: torch.Tensor, mask: torch.Tensor, fill: float = 0.0
) -> torch.Tensor:
    # Assert that the dimensions mask
    mask_dim = mask.shape
    tensor_dim = t.shape
    assert mask_dim == tensor_dim[: len(mask_dim)], (
        "mask dimension doesn't match tensor"
    )

    # Expand the mask tensor
    expand_dims = tensor_dim[len(mask_dim) :]
    for _ in expand_dims:
        mask = mask.unsqueeze(-1)

    # Fill where the mask is equal to 0
    t = t.masked_fill(mask == 0, fill)

    return t
