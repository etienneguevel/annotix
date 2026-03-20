import math

import torch
import torch.nn.functional as F

VALID_DISTORTIONS = {"identity", "cos", "revcos", "polyinc", "polydec"}


class TimeDistorter:
    """
    Warp time during training (to bias which timesteps the model sees)
    and during generation (to reshape the sampling schedule).

    Ported from DeFoG's ``flow_matching/time_distorter.py``, keeping only the
    closed-form distortions (identity, cos, revcos, polyinc, polydec).

    Args:
        train_distortion: Distortion applied when sampling training times.
        sample_distortion: Distortion applied to the generation schedule.
    """

    def __init__(
        self,
        train_distortion: str = "identity",
        sample_distortion: str = "identity",
    ):
        if train_distortion not in VALID_DISTORTIONS:
            raise ValueError(
                f"Unknown train_distortion '{train_distortion}'. "
                f"Valid: {sorted(VALID_DISTORTIONS)}"
            )
        if sample_distortion not in VALID_DISTORTIONS:
            raise ValueError(
                f"Unknown sample_distortion '{sample_distortion}'. "
                f"Valid: {sorted(VALID_DISTORTIONS)}"
            )
        self.train_distortion = train_distortion
        self.sample_distortion = sample_distortion

    def train_ft(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample distorted training times from U[0,1]. Returns (bs, 1)."""
        t = torch.rand((batch_size, 1), device=device)
        return self.apply_distortion(t, self.train_distortion)

    def sample_ft(self, t: torch.Tensor) -> torch.Tensor:
        """Apply sample distortion to a time tensor. Returns same shape."""
        return self.apply_distortion(t, self.sample_distortion)

    @staticmethod
    def apply_distortion(t: torch.Tensor, distortion_type: str) -> torch.Tensor:
        """Apply a closed-form time distortion to *t* in [0, 1]."""
        if distortion_type == "identity":
            return t
        elif distortion_type == "cos":
            return (1 - torch.cos(t * math.pi)) / 2
        elif distortion_type == "revcos":
            return 2 * t - (1 - torch.cos(t * math.pi)) / 2
        elif distortion_type == "polyinc":
            return t**2
        elif distortion_type == "polydec":
            return 2 * t - t**2
        else:
            raise ValueError(f"Unknown distortion type: {distortion_type}")


def p_xt_g_x1(
    X1: torch.Tensor,
    E1: torch.Tensor,
    t_float: torch.Tensor,
    limit_X: torch.Tensor,
    limit_E: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    DeFoG linear interpolation: p(x_t | x_1) = t · δ(x_1) + (1-t) · π_∞

    Args:
        X1: (bs, n) — integer labels for nodes
        E1: (bs, n, n) — integer labels for edges
        t_float: (bs, 1) — continuous time in [0, 1]
        limit_X: (natoms,) — marginal distribution over atom classes
        limit_E: (nbonds,) — marginal distribution over bond classes

    Returns:
        prob_X: (bs, n, natoms) — probability distribution over node classes
        prob_E: (bs, n, n, nbonds) — probability distribution over edge classes
    """
    device = X1.device
    limit_X = limit_X.to(device)
    limit_E = limit_E.to(device)

    t_node = t_float.view(-1, 1, 1)  # (bs, 1, 1)
    t_edge = t_float.view(-1, 1, 1, 1)  # (bs, 1, 1, 1)

    X1_onehot = F.one_hot(X1, num_classes=len(limit_X)).float()  # (bs, n, natoms)
    E1_onehot = F.one_hot(E1, num_classes=len(limit_E)).float()  # (bs, n, n, nbonds)

    prob_X = (
        t_node * X1_onehot + (1 - t_node) * limit_X[None, None, :]
    )  # (bs, n, natoms)
    prob_E = (
        t_edge * E1_onehot + (1 - t_edge) * limit_E[None, None, None, :]
    )  # (bs, n, n, nbonds)

    return prob_X.clamp(0.0, 1.0), prob_E.clamp(0.0, 1.0)


def dt_p_xt_g_x1(
    X1: torch.Tensor,
    E1: torch.Tensor,
    limit_X: torch.Tensor,
    limit_E: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Time derivative of p(x_t | x_1): d/dt p(x_t | x_1) = δ(x_1) - π_∞ (constant in t).

    Args:
        X1: (bs, n) — integer labels for nodes
        E1: (bs, n, n) — integer labels for edges
        limit_X: (natoms,) — marginal distribution over atom classes
        limit_E: (nbonds,) — marginal distribution over bond classes

    Returns:
        dX: (bs, n, natoms) — time derivative for nodes
        dE: (bs, n, n, nbonds) — time derivative for edges
    """
    device = X1.device
    limit_X = limit_X.to(device)
    limit_E = limit_E.to(device)

    X1_onehot = F.one_hot(X1, num_classes=len(limit_X)).float()  # (bs, n, natoms)
    E1_onehot = F.one_hot(E1, num_classes=len(limit_E)).float()  # (bs, n, n, nbonds)

    dX = X1_onehot - limit_X[None, None, :]  # (bs, n, natoms)
    dE = E1_onehot - limit_E[None, None, None, :]  # (bs, n, n, nbonds)

    return dX, dE


def sample_feature_noise(
    limit_X: torch.Tensor,
    limit_E: torch.Tensor,
    node_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample initial noise from the limit distribution (for generation start).

    Equivalent to DeFoG's flow_matching_utils.sample_discrete_feature_noise.
    Returns one-hot tensors with symmetric edges.

    Args:
        limit_X: (natoms,) — marginal distribution over atom classes
        limit_E: (nbonds,) — marginal distribution over bond classes
        node_mask: (bs, n_max) — binary mask for valid nodes

    Returns:
        U_X: (bs, n_max, natoms) — one-hot sampled node features
        U_E: (bs, n_max, n_max, nbonds) — one-hot symmetric sampled edge features
    """
    bs, n_max = node_mask.shape
    device = node_mask.device
    limit_X = limit_X.to(device)
    limit_E = limit_E.to(device)

    x_probs = limit_X[None, None, :].expand(bs, n_max, -1)  # (bs, n_max, natoms)
    e_probs = limit_E[None, None, None, :].expand(
        bs, n_max, n_max, -1
    )  # (bs, n_max, n_max, nbonds)

    U_X = x_probs.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max)  # (bs, n_max)
    U_E = (
        e_probs.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max, n_max)
    )  # (bs, n_max, n_max)

    U_X = F.one_hot(U_X, num_classes=len(limit_X)).float()  # (bs, n_max, natoms)
    U_E = F.one_hot(U_E, num_classes=len(limit_E)).float()  # (bs, n_max, n_max, nbonds)

    # Symmetrize: keep upper triangular part and mirror
    upper_mask = torch.zeros_like(U_E)
    indices = torch.triu_indices(n_max, n_max, offset=1)
    upper_mask[:, indices[0], indices[1], :] = 1.0
    U_E = U_E * upper_mask
    U_E = U_E + U_E.transpose(1, 2)

    return U_X, U_E
