import torch
import torch.nn as nn
from annotix_ml.graphtransf.data.data_utils import mask_any_tensor


def digress_loss(
    pN: torch.Tensor,
    pE: torch.Tensor,
    N: torch.Tensor,
    E: torch.Tensor,
    mask: torch.Tensor,
    loss_ratio: float,
) -> torch.Tensor:
    """
    Compute the loss based on prediction and target tensors.

    Args:
        pN (torch.Tensor): Predicted node probabilities of shape (bs, n, natoms).
        pE (torch.Tensor): Predicted edge probabilities of shape (bs, n, n, nedges).
        N (torch.Tensor): Target node features of shape (bs, n, natoms).
        E (torch.Tensor): Target edge features of shape (bs, n, n, nedges).
        mask (torch.Tensor): Mask tensor of shape (bs, n).
        loss_ratio (float): Weight of the edge loss in the total loss.

    Returns:
        torch.Tensor: Scalar total loss value.
    """
    ce_loss = nn.CrossEntropyLoss()

    # Node loss
    N_target = N.argmax(-1)  # (bs, n)
    N_target = mask_any_tensor(
        N_target, mask, fill=-100
    )  # -100 is the ignore_index of CrossEntropLoss
    pN = pN.transpose(1, 2)  # (bs, n_atoms, n)

    Nloss = ce_loss(pN, N_target)

    # Edges loss
    E_target = E.argmax(-1)  # (bs, n, n)
    E_target = mask_any_tensor(
        E_target, mask, fill=-100
    )  # -100 is the ignore_index of CrossEntropLoss
    pE = pE.permute((0, 3, 1, 2))  # (bs, n_edges, n, n)

    Eloss = ce_loss(pE, E_target)

    # Add the losses
    total_loss = Nloss + (loss_ratio * Eloss)

    return total_loss
