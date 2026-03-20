import torch
import torch.nn as nn
import torch.nn.functional as F

from annotix_ml.graphtransf.math.flow_matching import (
    dt_p_xt_g_x1,
    p_xt_g_x1,
)
from annotix_ml.graphtransf.math.noising import sample_discrete_features


class NoisingModelDefog(nn.Module):
    """
    DeFoG-style continuous-time discrete flow matching noising model.

    Drop-in replacement for NoisingModel that uses the linear flow:
        p(x_t | x_1) = t · δ(x_1) + (1-t) · π_∞
    where π_∞ is the limit distribution and t ∈ [0, 1] is continuous.

    Integer time steps [1, T] are stored externally; t_float = t / T is used internally.

    Args:
        nodes_distribution: (natoms,) — node type marginals (used for "marginal" transition).
        edges_distribution: (nbonds,) — edge type marginals (used for "marginal" transition).
        diffusion_steps: T — integer steps; t_float = integer_t / T.
        transition_type: limit distribution type — "uniform" | "marginal" | "absorbing".
    """

    def __init__(
        self,
        nodes_distribution: torch.Tensor,
        edges_distribution: torch.Tensor,
        diffusion_steps: int,
        transition_type: str = "uniform",
    ):
        super().__init__()
        self.T = diffusion_steps

        self.natoms, self.nbonds, limit_X, limit_E = self._build_limit_dist(
            transition_type, nodes_distribution, edges_distribution
        )
        self.limit_X = limit_X
        self.limit_E = limit_E

    @staticmethod
    def _build_limit_dist(
        transition_type: str,
        nodes_dist: torch.Tensor,
        edges_dist: torch.Tensor,
    ) -> tuple[int, int, torch.Tensor, torch.Tensor]:
        """Build the limit distribution π_∞ for nodes and edges."""
        natoms = len(nodes_dist)
        nbonds = len(edges_dist)

        if transition_type == "uniform":
            limit_X = torch.ones(natoms) / natoms
            limit_E = torch.ones(nbonds) / nbonds

        elif transition_type == "marginal":
            nd = nodes_dist.float()
            ed = edges_dist.float()
            limit_X = nd / nd.sum()
            limit_E = ed / ed.sum()

        elif transition_type == "absorbing":
            # Add a virtual absorbing class at the last position
            if natoms > 1:
                natoms += 1
            if nbonds > 1:
                nbonds += 1
            limit_X = torch.zeros(natoms)
            limit_X[-1] = 1.0
            limit_E = torch.zeros(nbonds)
            limit_E[-1] = 1.0

        else:
            raise ValueError(
                f"Unknown transition_type: {transition_type!r}. "
                "Expected 'uniform', 'marginal', or 'absorbing'."
            )

        return natoms, nbonds, limit_X, limit_E

    @torch.no_grad()
    def forward(
        self,
        N: torch.Tensor,
        E: torch.Tensor,
        node_mask: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Noise the nodes and edges of a batch to a random (or given) time step.

        Args:
            N: (bs, n, natoms) — one-hot node features.
            E: (bs, n, n, nbonds) — one-hot edge features.
            node_mask: (bs, n) — binary validity mask.
            t: optional integer time steps (bs,) or (bs, 1). Sampled from [1, T) if None.

        Returns:
            noised_N, noised_E, sampled_t — same shapes as NoisingModel.forward.
        """
        bs = N.shape[0]
        device = N.device

        if t is None:
            sampled_t = torch.randint(1, self.T, (bs,), device=device)
        else:
            sampled_t = t.to(device)
            if sampled_t.dim() == 2:
                sampled_t = sampled_t.squeeze(1)

        noised_N, noised_E = self.compute_noised_graph(N, E, sampled_t, node_mask)
        return noised_N, noised_E, sampled_t.unsqueeze(1)

    def compute_noised_graph(
        self,
        N: torch.Tensor,
        E: torch.Tensor,
        t: int | torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the noised graph at integer time step t via flow matching interpolation.

        Args:
            N: (bs, n, natoms) or (n, natoms) — one-hot node features.
            E: (bs, n, n, nbonds) or (n, n, nbonds) — one-hot edge features.
            t: integer time step(s). Converted to t_float = t / T internally.
            node_mask: (bs, n) — binary validity mask. Required for batched input.

        Returns:
            noised_N: (bs, n, natoms) — one-hot noised node features.
            noised_E: (bs, n, n, nbonds) — one-hot noised edge features.
        """
        is_unbatched = N.dim() == 2
        if is_unbatched:
            N = N.unsqueeze(0)
            E = E.unsqueeze(0)
            if node_mask is None:
                node_mask = torch.ones(N.shape[0], N.shape[1], device=N.device)
            else:
                node_mask = node_mask.unsqueeze(0)

        if node_mask is None:
            raise ValueError("node_mask must be provided.")

        bs = N.shape[0]
        device = N.device
        orig_dtype = N.dtype

        if isinstance(t, int):
            t_tensor = torch.full((bs,), t, dtype=torch.float, device=device)
        else:
            t_tensor = t.to(device).float()
            if t_tensor.dim() == 0:
                t_tensor = t_tensor.unsqueeze(0)

        t_float = (t_tensor / self.T).unsqueeze(-1)  # (bs, 1)

        # Integer class labels from one-hot
        X1_label = N.argmax(-1)  # (bs, n)
        E1_label = E.argmax(-1)  # (bs, n, n)

        # Flow matching interpolation
        prob_N, prob_E = p_xt_g_x1(
            X1_label, E1_label, t_float, self.limit_X, self.limit_E
        )  # (bs, n, natoms), (bs, n, n, nbonds)

        # Sample one-hot discrete features
        N_noised, E_noised = sample_discrete_features(prob_N, prob_E, node_mask)

        N_noised = N_noised.to(orig_dtype)
        E_noised = E_noised.to(orig_dtype)

        if is_unbatched:
            return N_noised.squeeze(0), E_noised.squeeze(0)

        return N_noised, E_noised

    def get_posterior(
        self,
        N: torch.Tensor,
        E: torch.Tensor,
        t: int,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute p(x_s | x_t, x_1) under the linear flow.

        In the linear flow, x_s is conditionally independent of x_t given x_1:
            p(x_s = j | x_t, x_1 = k) = p(x_s = j | x_1 = k)
                                       = s · δ(k, j) + (1-s) · π_∞(j)
        where s = (t-1) / T.

        N and E are only used for shape / device; the posterior does not depend on x_t.

        Args:
            N: (bs, n, natoms) — current noisy node features (for shape/device).
            E: (bs, n, n, nbonds) — current noisy edge features (for shape/device).
            t: integer time step (current step; target step is t-1).
            eps: unused, kept for API compatibility with NoisingModel.

        Returns:
            pN_posterior: (bs, n, natoms, natoms) — [b, i, k, j] = p(x_s=j | x_1=k)
            pE_posterior: (bs, n, n, nbonds, nbonds)
        """
        bs, n, _ = N.shape
        device = N.device

        s = (t - 1) / self.T  # continuous time of the target step

        limit_X = self.limit_X.to(device)
        limit_E = self.limit_E.to(device)

        # Build (natoms, natoms): row k = p(x_s | x_1=k) = s·I[k,:] + (1-s)·π_∞
        I_N = torch.eye(self.natoms, device=device)  # (natoms, natoms)
        limit_rows_N = limit_X.unsqueeze(0).expand(self.natoms, -1)  # (natoms, natoms)
        pN_post_2d = s * I_N + (1 - s) * limit_rows_N  # (natoms, natoms)

        # Expand to (bs, n, natoms, natoms)
        pN_posterior = pN_post_2d.unsqueeze(0).unsqueeze(0).expand(bs, n, -1, -1)

        # Build (nbonds, nbonds) posterior for edges
        I_E = torch.eye(self.nbonds, device=device)  # (nbonds, nbonds)
        limit_rows_E = limit_E.unsqueeze(0).expand(self.nbonds, -1)  # (nbonds, nbonds)
        pE_post_2d = s * I_E + (1 - s) * limit_rows_E  # (nbonds, nbonds)

        # Expand to (bs, n, n, nbonds, nbonds)
        pE_posterior = (
            pE_post_2d.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(bs, n, n, -1, -1)
        )

        return pN_posterior, pE_posterior

    def move_to(self, device: torch.device) -> None:
        """Move limit distribution tensors to the specified device."""
        self.limit_X = self.limit_X.to(device)
        self.limit_E = self.limit_E.to(device)

    def compute_rate_matrix(
        self,
        t_float: torch.Tensor,
        node_mask: torch.Tensor,
        G_t: tuple[torch.Tensor, torch.Tensor],
        G_1_pred: tuple[torch.Tensor, torch.Tensor],
        rdb: str = "general",
        rdb_crit: str = "x_1",
        eta: float = 0.0,
        omega: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the full rate matrix R_t = R*_t + R^db_t + R^tg_t.

        Args:
            t_float: (bs, 1) — continuous time in [0, 1].
            node_mask: (bs, n) — binary validity mask.
            G_t: (X_t, E_t) — one-hot tensors of the current noisy graph.
            G_1_pred: (X_1_pred, E_1_pred) — predicted probabilities over the clean graph.
            rdb: rate-matrix design choice — "general" | "marginal" | "column".
            rdb_crit: column-selection criterion (used when rdb="column").
            eta: stochasticity weight for R^db.
            omega: target-guidance weight for R^tg.

        Returns:
            R_t_X: (bs, n, natoms) — node rate matrix.
            R_t_E: (bs, n, n, nbonds) — edge rate matrix.
        """
        X_t, E_t = G_t
        X_1_pred, E_1_pred = G_1_pred

        X_t_label = X_t.argmax(-1, keepdim=True)  # (bs, n, 1)
        E_t_label = E_t.argmax(-1, keepdim=True)  # (bs, n, n, 1)

        # Sample clean graph from predictions using the existing utility
        X_1_oh, E_1_oh = sample_discrete_features(X_1_pred, E_1_pred, node_mask)
        X_1_sampled = X_1_oh.argmax(-1)  # (bs, n)
        E_1_sampled = E_1_oh.argmax(-1)  # (bs, n, n)

        dfm_variables = self.compute_dfm_variables(
            t_float, X_t_label, E_t_label, X_1_sampled, E_1_sampled
        )

        Rstar_t_X, Rstar_t_E = self.compute_Rstar(dfm_variables)
        Rdb_t_X, Rdb_t_E = self.compute_RDB(
            X_t_label,
            E_t_label,
            X_1_pred,
            E_1_pred,
            X_1_sampled,
            E_1_sampled,
            node_mask,
            t_float,
            dfm_variables,
            rdb,
            rdb_crit,
            eta,
        )
        Rtg_t_X, Rtg_t_E = self.compute_R_tg(
            X_1_sampled, E_1_sampled, X_t_label, E_t_label, dfm_variables, omega
        )

        R_t_X = Rstar_t_X + Rdb_t_X + Rtg_t_X
        R_t_E = Rstar_t_E + Rdb_t_E + Rtg_t_E

        R_t_X, R_t_E = self.stabilize_rate_matrix(R_t_X, R_t_E, dfm_variables)

        return R_t_X, R_t_E

    def compute_dfm_variables(
        self,
        t_float: torch.Tensor,
        X_t_label: torch.Tensor,
        E_t_label: torch.Tensor,
        X_1_sampled: torch.Tensor,
        E_1_sampled: torch.Tensor,
    ) -> dict:
        """
        Compute intermediate DFM variables needed by the rate matrix components.

        Args:
            t_float: (bs, 1) — continuous time in [0, 1].
            X_t_label: (bs, n, 1) — current node class indices (keepdim).
            E_t_label: (bs, n, n, 1) — current edge class indices (keepdim).
            X_1_sampled: (bs, n) — sampled clean node class indices.
            E_1_sampled: (bs, n, n) — sampled clean edge class indices.

        Returns:
            dict with keys: pt_vals_X, pt_vals_E, pt_vals_at_Xt, pt_vals_at_Et,
                            dt_p_vals_X, dt_p_vals_E, dt_p_vals_at_Xt, dt_p_vals_at_Et,
                            Z_t_X, Z_t_E.
        """
        dt_p_vals_X, dt_p_vals_E = dt_p_xt_g_x1(
            X_1_sampled, E_1_sampled, self.limit_X, self.limit_E
        )  # (bs, n, natoms), (bs, n, n, nbonds)

        dt_p_vals_at_Xt = dt_p_vals_X.gather(-1, X_t_label).squeeze(-1)  # (bs, n)
        dt_p_vals_at_Et = dt_p_vals_E.gather(-1, E_t_label).squeeze(-1)  # (bs, n, n)

        pt_vals_X, pt_vals_E = p_xt_g_x1(
            X_1_sampled, E_1_sampled, t_float, self.limit_X, self.limit_E
        )  # (bs, n, natoms), (bs, n, n, nbonds)

        pt_vals_at_Xt = pt_vals_X.gather(-1, X_t_label).squeeze(-1)  # (bs, n)
        pt_vals_at_Et = pt_vals_E.gather(-1, E_t_label).squeeze(-1)  # (bs, n, n)

        Z_t_X = torch.count_nonzero(pt_vals_X, dim=-1)  # (bs, n)
        Z_t_E = torch.count_nonzero(pt_vals_E, dim=-1)  # (bs, n, n)

        return {
            "pt_vals_X": pt_vals_X,
            "pt_vals_E": pt_vals_E,
            "pt_vals_at_Xt": pt_vals_at_Xt,
            "pt_vals_at_Et": pt_vals_at_Et,
            "dt_p_vals_X": dt_p_vals_X,
            "dt_p_vals_E": dt_p_vals_E,
            "dt_p_vals_at_Xt": dt_p_vals_at_Xt,
            "dt_p_vals_at_Et": dt_p_vals_at_Et,
            "Z_t_X": Z_t_X,
            "Z_t_E": Z_t_E,
        }

    def compute_Rstar(self, dfm_variables: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the minimal stochastic rate matrix R*_t.

        Returns:
            Rstar_t_X: (bs, n, natoms)
            Rstar_t_E: (bs, n, n, nbonds)
        """
        dt_p_vals_X = dfm_variables["dt_p_vals_X"]
        dt_p_vals_E = dfm_variables["dt_p_vals_E"]
        dt_p_vals_at_Xt = dfm_variables["dt_p_vals_at_Xt"]
        dt_p_vals_at_Et = dfm_variables["dt_p_vals_at_Et"]
        pt_vals_at_Xt = dfm_variables["pt_vals_at_Xt"]
        pt_vals_at_Et = dfm_variables["pt_vals_at_Et"]
        Z_t_X = dfm_variables["Z_t_X"]
        Z_t_E = dfm_variables["Z_t_E"]

        inner_X = dt_p_vals_X - dt_p_vals_at_Xt[:, :, None]
        inner_E = dt_p_vals_E - dt_p_vals_at_Et[:, :, :, None]
        numer_X = F.relu(inner_X)  # (bs, n, natoms)
        numer_E = F.relu(inner_E)  # (bs, n, n, nbonds)

        denom_X = Z_t_X * pt_vals_at_Xt  # (bs, n)
        denom_E = Z_t_E * pt_vals_at_Et  # (bs, n, n)

        Rstar_t_X = numer_X / denom_X[:, :, None]
        Rstar_t_E = numer_E / denom_E[:, :, :, None]

        return Rstar_t_X, Rstar_t_E

    def compute_RDB(
        self,
        X_t_label: torch.Tensor,
        E_t_label: torch.Tensor,
        X_1_pred: torch.Tensor,
        E_1_pred: torch.Tensor,
        X_1_sampled: torch.Tensor,
        E_1_sampled: torch.Tensor,
        node_mask: torch.Tensor,
        t_float: torch.Tensor,
        dfm_variables: dict,
        rdb: str,
        rdb_crit: str,
        eta: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the design-based rate matrix component R^db_t.

        Args:
            X_t_label: (bs, n, 1) — current node class indices (keepdim).
            E_t_label: (bs, n, n, 1) — current edge class indices (keepdim).
            X_1_pred: (bs, n, natoms) — predicted node class probabilities.
            E_1_pred: (bs, n, n, nbonds) — predicted edge class probabilities.
            X_1_sampled: (bs, n) — sampled clean node class indices.
            E_1_sampled: (bs, n, n) — sampled clean edge class indices.
            node_mask: (bs, n) — binary validity mask.
            t_float: (bs, 1) — continuous time in [0, 1].
            dfm_variables: precomputed DFM variables.
            rdb: rate-matrix design — "general" | "marginal" | "column".
            rdb_crit: column criterion used when rdb="column".
            eta: stochasticity weight.

        Returns:
            Rdb_t_X: (bs, n, natoms)
            Rdb_t_E: (bs, n, n, nbonds)
        """
        pt_vals_X = dfm_variables["pt_vals_X"]
        pt_vals_E = dfm_variables["pt_vals_E"]

        dx = pt_vals_X.shape[-1]
        de = pt_vals_E.shape[-1]

        if rdb == "general":
            x_mask = torch.ones_like(pt_vals_X)
            e_mask = torch.ones_like(pt_vals_E)

        elif rdb == "marginal":
            limit_X = self.limit_X.to(X_t_label.device)
            limit_E = self.limit_E.to(E_t_label.device)

            Xt_marginal = limit_X[X_t_label]
            Et_marginal = limit_E[E_t_label]

            x_mask = limit_X.expand(X_t_label.shape[0], X_t_label.shape[1], -1)
            e_mask = limit_E.expand(
                E_t_label.shape[0], E_t_label.shape[1], E_t_label.shape[2], -1
            )
            x_mask = x_mask > Xt_marginal
            e_mask = e_mask > Et_marginal

        elif rdb == "column":
            if rdb_crit == "max_marginal":
                x_col = self.limit_X.argmax(keepdim=True).expand(X_t_label.shape)
                e_col = self.limit_E.argmax(keepdim=True).expand(E_t_label.shape)

            elif rdb_crit == "x_t":
                x_col = X_t_label
                e_col = E_t_label

            elif rdb_crit == "abs_state":
                x_col = torch.ones_like(X_t_label) * (dx - 1)
                e_col = torch.ones_like(E_t_label) * (de - 1)

            elif rdb_crit == "p_x1_g_xt":
                x_col = X_1_pred.argmax(dim=-1, keepdim=True)
                e_col = E_1_pred.argmax(dim=-1, keepdim=True)

            elif rdb_crit == "x_1":
                x_col = X_1_sampled.unsqueeze(-1)
                e_col = E_1_sampled.unsqueeze(-1)

            elif rdb_crit == "p_xt_g_x1":
                x_col = pt_vals_X.argmax(dim=-1, keepdim=True)
                e_col = pt_vals_E.argmax(dim=-1, keepdim=True)

            elif rdb_crit == "xhat_t":
                X_hat_oh, E_hat_oh = sample_discrete_features(
                    pt_vals_X.clone(), pt_vals_E.clone(), node_mask
                )
                x_col = X_hat_oh.argmax(-1).unsqueeze(-1)
                e_col = E_hat_oh.argmax(-1).unsqueeze(-1)

            else:
                raise NotImplementedError(f"rdb_crit={rdb_crit!r} is not implemented.")

            x_mask = F.one_hot(x_col.squeeze(-1), num_classes=dx).float()
            x_mask[(x_col == X_t_label).squeeze(-1)] = 1.0
            e_mask = F.one_hot(e_col.squeeze(-1), num_classes=de).float()
            e_mask[(e_col == E_t_label).squeeze(-1)] = 1.0

        else:
            raise NotImplementedError(f"rdb={rdb!r} is not implemented.")

        Rdb_t_X = pt_vals_X * x_mask * eta
        Rdb_t_E = pt_vals_E * e_mask * eta

        return Rdb_t_X, Rdb_t_E

    def compute_R_tg(
        self,
        X_1_sampled: torch.Tensor,
        E_1_sampled: torch.Tensor,
        X_t_label: torch.Tensor,
        E_t_label: torch.Tensor,
        dfm_variables: dict,
        omega: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the target-guidance rate matrix component R^tg_t.

        Args:
            X_1_sampled: (bs, n) — sampled clean node class indices.
            E_1_sampled: (bs, n, n) — sampled clean edge class indices.
            X_t_label: (bs, n, 1) — current node class indices (keepdim).
            E_t_label: (bs, n, n, 1) — current edge class indices (keepdim).
            dfm_variables: precomputed DFM variables.
            omega: target-guidance weight.

        Returns:
            Rtg_t_X: (bs, n, natoms)
            Rtg_t_E: (bs, n, n, nbonds)
        """
        pt_vals_at_Xt = dfm_variables["pt_vals_at_Xt"]
        pt_vals_at_Et = dfm_variables["pt_vals_at_Et"]
        Z_t_X = dfm_variables["Z_t_X"]
        Z_t_E = dfm_variables["Z_t_E"]

        X1_onehot = F.one_hot(X_1_sampled, num_classes=self.natoms).float()
        E1_onehot = F.one_hot(E_1_sampled, num_classes=self.nbonds).float()

        mask_X = X_1_sampled.unsqueeze(-1) != X_t_label  # (bs, n, 1) broadcast
        mask_E = E_1_sampled.unsqueeze(-1) != E_t_label  # (bs, n, n, 1) broadcast

        numer_X = X1_onehot * omega * mask_X
        numer_E = E1_onehot * omega * mask_E

        denom_X = Z_t_X * pt_vals_at_Xt  # (bs, n)
        denom_E = Z_t_E * pt_vals_at_Et  # (bs, n, n)

        Rtg_t_X = numer_X / denom_X[:, :, None]
        Rtg_t_E = numer_E / denom_E[:, :, :, None]

        return Rtg_t_X, Rtg_t_E

    def stabilize_rate_matrix(
        self,
        R_t_X: torch.Tensor,
        R_t_E: torch.Tensor,
        dfm_variables: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Stabilize rate matrices: clamp NaN/Inf and zero out impossible transitions.

        Args:
            R_t_X: (bs, n, natoms) — node rate matrix.
            R_t_E: (bs, n, n, nbonds) — edge rate matrix.
            dfm_variables: precomputed DFM variables.

        Returns:
            Stabilized (R_t_X, R_t_E).
        """
        pt_vals_X = dfm_variables["pt_vals_X"]
        pt_vals_E = dfm_variables["pt_vals_E"]
        pt_vals_at_Xt = dfm_variables["pt_vals_at_Xt"]
        pt_vals_at_Et = dfm_variables["pt_vals_at_Et"]

        R_t_X = torch.nan_to_num(R_t_X, nan=0.0, posinf=0.0, neginf=0.0)
        R_t_E = torch.nan_to_num(R_t_E, nan=0.0, posinf=0.0, neginf=0.0)
        R_t_X[R_t_X > 1e5] = 0.0
        R_t_E[R_t_E > 1e5] = 0.0

        dx = R_t_X.shape[-1]
        de = R_t_E.shape[-1]

        # Zero out rows where current state has zero probability
        R_t_X[(pt_vals_at_Xt == 0.0)[:, :, None].expand(-1, -1, dx)] = 0.0
        R_t_E[(pt_vals_at_Et == 0.0)[:, :, :, None].expand(-1, -1, -1, de)] = 0.0

        # Zero out columns with zero probability under the flow
        R_t_X[pt_vals_X == 0.0] = 0.0
        R_t_E[pt_vals_E == 0.0] = 0.0

        return R_t_X, R_t_E
