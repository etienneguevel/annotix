import torch
from omegaconf import DictConfig
from torch.linalg import LinAlgError
from tqdm import tqdm

from annotix_ml.data.atoms_data import VALID_ELEMENTS
from annotix_ml.data.data_utils import mask_any_tensor
from annotix_ml.distributed import get_global_rank, get_global_size
from annotix_ml.graphtransf.arch.digress_meta_arch import DigressMetaArch
from annotix_ml.graphtransf.math.flow_matching import (
    TimeDistorter,
    sample_feature_noise,
)
from annotix_ml.graphtransf.math.losses import digress_loss
from annotix_ml.graphtransf.math.noising import sample_discrete_features
from annotix_ml.graphtransf.models.noising_defog import NoisingModelDefog


class DefogMetaArch(DigressMetaArch):
    """
    DeFoG-style architecture using continuous-time flow matching.

    Inherits all training methods from DigressMetaArch unchanged. Only `__init__`
    and `generate` are overridden to use NoisingModelDefog and the forward
    generation loop (t=0 → sample_steps).

    Args:
        transition_type (str | None): Limit distribution type — "uniform" | "marginal".
            If None, inferred from noise_strategy ("uniform" → "uniform", else "marginal").
        sample_steps (int | None): Number of generation steps. If None, uses diffusion_steps.
        rdb (str): Rate-matrix design — "general" | "marginal" | "column".
        rdb_crit (str): Column criterion (used when rdb="column").
        eta (float): Stochasticity weight for R^db.
        omega (float): Target-guidance weight for R^tg.

    Note: "absorbing" transition_type is not supported — it adds virtual classes that
    mismatch the GNN's output dimensions.
    """

    def __init__(
        self,
        d,
        de,
        dy,
        n_heads,
        n_layers,
        noise_strategy,
        diffusion_steps,
        loss_ratio,
        device,
        valid_elements,
        y_update=True,
        no_y=False,
        nodes_distribution=None,
        edges_distribution=None,
        k=None,
        extra_features=None,
        max_weight=None,
        morgan_nbits=2048,
        encoder_output_size=None,
        fp_projector_type="linear",
        mixed_precision=True,
        rdb="general",
        rdb_crit="x_1",
        eta=0.0,
        omega=0.0,
        train_distortion="identity",
        sample_distortion="identity",
    ):
        super().__init__(
            d=d,
            de=de,
            dy=dy,
            n_heads=n_heads,
            n_layers=n_layers,
            noise_strategy=noise_strategy,
            diffusion_steps=diffusion_steps,
            loss_ratio=loss_ratio,
            device=device,
            valid_elements=valid_elements,
            y_update=y_update,
            no_y=no_y,
            nodes_distribution=nodes_distribution,
            edges_distribution=edges_distribution,
            k=k,
            extra_features=extra_features,
            max_weight=max_weight,
            morgan_nbits=morgan_nbits,
            encoder_output_size=encoder_output_size,
            fp_projector_type=fp_projector_type,
            mixed_precision=mixed_precision,
        )

        if noise_strategy is None:
            noise_strategy = "uniform" if noise_strategy == "uniform" else "marginal"

        self.noiser = NoisingModelDefog(
            nodes_distribution=self.nodes_distribution,
            edges_distribution=self.edges_distribution,
            diffusion_steps=diffusion_steps,
            transition_type=noise_strategy,
        )
        self.noiser.move_to(device)

        self.sample_steps = diffusion_steps
        self.rdb = rdb
        self.rdb_crit = rdb_crit
        self.eta = eta
        self.omega = omega
        self.time_distorter = TimeDistorter(train_distortion, sample_distortion)

    @classmethod
    def init_from_cfg(
        cls,
        cfg: DictConfig,
        device: torch.device,
        valid_elements,
        nodes_distribution,
        edges_distribution,
        max_weight=None,
    ):
        if not valid_elements:
            valid_elements = list(VALID_ELEMENTS)

        morgan_nbits = cfg.dataset.get("morgan_nbits") or 0
        encoder_output_size = (
            cfg.spectra_encoder.get("output_size")
            if cfg.get("spectra_encoder") is not None
            else None
        )

        return cls(
            d=cfg.model.d,
            de=cfg.model.de,
            dy=cfg.model.dy,
            n_heads=cfg.model.n_heads,
            n_layers=cfg.model.n_layers,
            y_update=cfg.model.y_update,
            no_y=cfg.model.no_y,
            noise_strategy=cfg.model.noise_strategy,
            diffusion_steps=cfg.model.diffusion_steps,
            loss_ratio=cfg.train.loss_ratio,
            device=device,
            k=cfg.model.num_ev,
            extra_features=cfg.model.extra_features,
            valid_elements=valid_elements,
            nodes_distribution=nodes_distribution,
            edges_distribution=edges_distribution,
            max_weight=max_weight,
            morgan_nbits=morgan_nbits,
            encoder_output_size=encoder_output_size,
            fp_projector_type=cfg.model.get("fp_projector_type", "linear"),
            mixed_precision=cfg.train.get("mixed_precision", True),
            rdb=cfg.model.get("rdb", "general"),
            rdb_crit=cfg.model.get("rdb_crit", "x_1"),
            eta=cfg.model.get("eta", 0.0),
            omega=cfg.model.get("omega", 0.0),
            train_distortion=cfg.model.get("train_distortion", "identity"),
            sample_distortion=cfg.model.get("sample_distortion", "identity"),
        )

    def forward_backward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        mask: torch.Tensor,
        fp: torch.Tensor | None = None,
        loss_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Override parent to sample time via ``TimeDistorter`` instead of uniform.

        The distorted time is passed explicitly to the noiser so that the model
        sees more of certain timesteps during training (e.g. cos distortion
        spends more time near t=0 and t=1).
        """
        bs = nodes.shape[0]

        # Sample distorted time and convert to integer steps for the noiser
        t_float = self.time_distorter.train_ft(bs, nodes.device)  # (bs, 1)
        t_int = (t_float * self.noiser.T).long().clamp(1, self.noiser.T - 1)

        N_noised, E_noised, sampled_t = self.noiser(nodes, edges, mask, t=t_int)
        pos_emb, y = self.compute_extra_features(N_noised, E_noised, mask, sampled_t)
        if fp is not None:
            if self.fp_projector is not None:
                fp = self.fp_projector(fp)
            y = torch.cat([y, fp], dim=-1)

        input_args = (E_noised, mask, y, pos_emb, N_noised)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=self.mixed_precision
        ):
            out = self.diffuser(*input_args)
            pN, pE, *_ = out
            loss = digress_loss(pN, pE, nodes, edges, mask, self.loss_ratio)

        self.scaler.scale(loss * loss_scale).backward()

        return pN, pE, loss

    @torch.no_grad()
    def generate(
        self,
        num_samples,
        max_nodes,
        min_nodes=6,
        progress_bar=False,
        num_attempts=3,
        fp=None,
    ):
        """
        Generate graphs using DeFoG's forward flow matching loop (t=0 → sample_steps).

        At each step, the model predicts the clean graph, a rate matrix is computed,
        and the next state is sampled from the resulting step probabilities.

        Args:
            num_samples (int | torch.Tensor): Number of samples or a 1-D tensor of node counts.
            max_nodes (int): Maximum number of nodes per graph.
            min_nodes (int): Minimum number of nodes per graph. Defaults to 6.
            progress_bar (bool): Show tqdm progress bar. Defaults to False.
            num_attempts (int): Retry attempts on LinAlgError. Defaults to 3.
            fp (torch.Tensor | None): Pre-computed fingerprint of shape (bs, morgan_nbits).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (N, E, mask)
        """
        if isinstance(num_samples, torch.Tensor):
            assert len(num_samples.shape) == 1, (
                f"Wrong shape for num_samples: {num_samples.shape}"
            )
            n_sizes = num_samples
            num_samples = n_sizes.shape[0]
        else:
            n_sizes = None

        i = 0
        while i < num_attempts:
            n = (
                n_sizes
                if n_sizes is not None
                else torch.randint(min_nodes, max_nodes, (num_samples,))
            )

            mask = torch.stack(
                [
                    torch.cat([torch.ones(int(m)), torch.zeros(max_nodes - int(m))])
                    for m in n
                ]
            ).to(self.device)

            N, E = sample_feature_noise(self.noiser.limit_X, self.noiser.limit_E, mask)
            N = N.float().to(self.device)
            E = E.float().to(self.device)
            N = mask_any_tensor(N, mask, fill=0)
            E = mask_any_tensor(E, mask, fill=0)

            self.diffuser.eval()

            if progress_bar:
                t_range = tqdm(
                    range(self.sample_steps),
                    total=self.sample_steps,
                    desc="Generating",
                    position=get_global_rank() + get_global_size(),
                )
            else:
                t_range = range(self.sample_steps)

            try:
                for t_int in t_range:
                    t_raw = torch.tensor(
                        [t_int / self.sample_steps], device=self.device
                    )
                    s_raw = torch.tensor(
                        [(t_int + 1) / self.sample_steps], device=self.device
                    )

                    t_distorted = self.time_distorter.sample_ft(t_raw).item()
                    s_distorted = self.time_distorter.sample_ft(s_raw).item()
                    dt = max(s_distorted - t_distorted, 1e-6)

                    N = N.float().to(self.device)
                    E = E.float().to(self.device)
                    N = mask_any_tensor(N, mask, fill=0)
                    E = mask_any_tensor(E, mask, fill=0)

                    t_cond = torch.full(
                        (num_samples, 1),
                        t_distorted * self.noiser.T,
                        device=self.device,
                    )

                    pN, pE = self.forward(N, E, mask, t_cond, fp)
                    pN = pN.softmax(-1)
                    pE = pE.softmax(-1)

                    if s_distorted >= 1.0:
                        N, E = sample_discrete_features(pN, pE, mask)
                        break

                    t_float_tensor = torch.full(
                        (num_samples, 1), t_distorted, device=self.device
                    )
                    R_t_X, R_t_E = self.noiser.compute_rate_matrix(
                        t_float_tensor,
                        mask,
                        (N, E),
                        (pN, pE),
                        rdb=self.rdb,
                        rdb_crit=self.rdb_crit,
                        eta=self.eta,
                        omega=self.omega,
                    )

                    step_probs_X, step_probs_E = self._compute_step_probs(
                        R_t_X, R_t_E, N, E, dt
                    )
                    N, E = sample_discrete_features(step_probs_X, step_probs_E, mask)

                return N.float(), E.float(), mask

            except LinAlgError as e:
                print(
                    f"Generation failed, attempt {i + 1} / {num_attempts}: "
                    f"{type(e).__name__}: {e}"
                )
                i += 1

    @staticmethod
    def _compute_step_probs(
        R_t_X: torch.Tensor,
        R_t_E: torch.Tensor,
        X_t: torch.Tensor,
        E_t: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert rate matrices to discrete step transition probabilities.

        Off-diagonal entries: prob = R_t * dt.
        Diagonal entries: 1 - sum(off-diagonal), clamped >= 0.

        Args:
            R_t_X: (bs, n, natoms) — node rate matrix.
            R_t_E: (bs, n, n, nbonds) — edge rate matrix.
            X_t: (bs, n, natoms) — current one-hot node features.
            E_t: (bs, n, n, nbonds) — current one-hot edge features.
            dt: float — time step size (1 / sample_steps).

        Returns:
            step_probs_X: (bs, n, natoms)
            step_probs_E: (bs, n, n, nbonds)
        """
        step_probs_X = R_t_X * dt
        step_probs_E = R_t_E * dt

        x_labels = X_t.argmax(-1, keepdim=True)
        e_labels = E_t.argmax(-1, keepdim=True)

        step_probs_X.scatter_(-1, x_labels, 0.0)
        step_probs_E.scatter_(-1, e_labels, 0.0)

        step_probs_X.scatter_(
            -1,
            x_labels,
            (1.0 - step_probs_X.sum(dim=-1, keepdim=True)).clamp(min=0.0),
        )
        step_probs_E.scatter_(
            -1,
            e_labels,
            (1.0 - step_probs_E.sum(dim=-1, keepdim=True)).clamp(min=0.0),
        )

        return step_probs_X.clone(), step_probs_E.clone()
