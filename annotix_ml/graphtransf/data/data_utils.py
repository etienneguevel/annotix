import torch


def mask_any_tensor(
    t: torch.Tensor, mask: torch.Tensor, fill: float = 0.0
) -> torch.Tensor:
    """Apply a 0/1 mask to the leading dimensions of a tensor, filling masked
    positions with a scalar value.

    The function expects ``mask`` to have shape equal to the leading (left-most)
    dimensions of ``t``. Any remaining trailing dimensions of ``t`` are treated
    as feature dimensions and the mask is expanded (unsqueezed) over them.

    Behavior summary
    - If ``mask.shape == t.shape[:len(mask.shape)]`` the mask is unsqueezed
      for each remaining trailing dimension of ``t`` and then applied.
    - Positions where ``mask == 0`` are replaced by ``fill`` via
      :meth:`torch.Tensor.masked_fill`.
    - Non-zero mask values are treated as keep indicators.

    Args:
        t: Input tensor to mask. Any dtype is supported; ``fill`` will be
           cast as needed by PyTorch when filling.
        mask: Binary (0/1) or boolean tensor whose shape must match the leading
           dimensions of ``t``. For example, if ``t`` has shape
           ``(B, N, F)`` then ``mask`` can be ``(B, N)`` or ``(B, N, 1)``.
        fill: Scalar value used to fill masked locations (where ``mask == 0``).

    Returns:
        A tensor of the same shape and dtype as ``t`` with masked positions
        replaced by ``fill``.

    Example:
        >>> t = torch.ones(2, 3, 4)
        >>> mask = torch.tensor([[1, 0, 1], [0, 1, 1]])  # shape (2, 3)
        >>> out = mask_any_tensor(t, mask, fill=0.0)
        >>> out.shape
        torch.Size([2, 3, 4])

    Notes and edge cases:
    - The function asserts that ``mask.shape`` equals the leading dimensions of
      ``t``; a mismatch will raise an AssertionError.
    - ``mask`` may be integer or boolean. Zeros are considered masked.
    - Time/space complexity is linear in the number of elements in ``t``.

    """
    # Assert that the dimensions match
    mask_dim = mask.shape
    tensor_dim = t.shape
    assert mask_dim == tensor_dim[: len(mask_dim)], (
        "mask dimension doesn't match tensor",
        mask_dim,
        tensor_dim,
    )

    # Expand the mask tensor
    expand_dims = tensor_dim[len(mask_dim) :]
    for _ in expand_dims:
        mask = mask.unsqueeze(-1)

    # Fill where the mask is equal to 0
    t = t.masked_fill(mask == 0, fill)

    return t
