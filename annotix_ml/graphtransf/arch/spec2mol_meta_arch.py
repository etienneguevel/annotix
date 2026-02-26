from typing import Literal
import torch
import torch.nn as nn
from omegaconf import DictConfig

from annotix_ml import BASE_DIR
from annotix_ml.graphtransf.arch import DigressMetaArch
from annotix_ml.spectraencoder.model.spectra_encoder import SpectraEncoderGrowing
from annotix_ml.graphtransf.math.extra_features import spectra_fingerprint


class Spec2MolMetaArch(DigressMetaArch):
    """
    Architecture for the Spec2Mol model, extending DiGress with spectral features.

    This class handles the initialization of the diffusion model (GNN) and the noising model,
    as well as the computation of extra features (including spectral fingerprints)
    and the forward pass for training and generation.

    Attributes:
        known_extra_features (list[str]): List of supported extra feature names.
    """

    known_extra_features = [
        "laplacian_embedding",
        "node_cycle",
        "valence_features",
        "spectra_fingerprint",
    ]

    def __init__(
        self,
        d: int,
        de: int,
        dy: int,
        n_heads: int,
        n_layers: int,
        noise_strategy: Literal["uniform", "distribution"],
        diffusion_steps: int,
        loss_ratio: float,
        device: torch.device,
        valid_elements: list[str],
        y_update: bool = True,
        no_y: bool = False,
        nodes_distribution: torch.Tensor | None = None,
        edges_distribution: torch.Tensor | None = None,
        k: int | None = None,
        extra_features: list[str] | None = None,
        max_weight: float | None = None,
        morgan_nbits: int = 2048,
        # SpectraEncoder args
        form_embedder: str = "float",
        output_size: int = 4096,
        hidden_size: int = 512,
        spectra_dropout: float = 0.0,
        top_layers: int = 1,
        refine_layers: int = 4,
        magma_modulo: int = 2048,
        peak_attn_layers: int = 2,
        set_pooling: str = "intensity",
        pairwise_featurization: bool = True,
        num_heads: int = 8,
        embed_instrument: bool = False,
        inten_transform: str = "float",
        no_diffs: bool = False,
    ):
        # Store info specific to spectra before super().__init__ (which calls properties)
        self.morgan_nbits = morgan_nbits

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
        )

        # Build the models for spectra treatment
        self.spectra_encoder = SpectraEncoderGrowing(
            form_embedder=form_embedder,
            output_size=output_size,
            hidden_size=hidden_size,
            spectra_dropout=spectra_dropout,
            top_layers=top_layers,
            magma_modulo=magma_modulo,
            peak_attn_layers=peak_attn_layers,
            set_pooling=set_pooling,
            pairwise_featurization=pairwise_featurization,
            num_heads=num_heads,
            embed_instrument=embed_instrument,
            inten_transform=inten_transform,
            no_diffs=no_diffs,
            refine_layers=refine_layers,
        )

        self.merge_function = nn.Linear(output_size, morgan_nbits)

    @classmethod
    def init_from_cfg(
        cls,
        cfg: DictConfig,
        device: torch.device,
        valid_elements: list[str] | None,
        nodes_distribution: torch.Tensor | None,
        edges_distribution: torch.Tensor | None,
        max_weight: float | None = None,
    ):
        arch = super().init_from_cfg(
            cfg,
            device,
            valid_elements,
            nodes_distribution,
            edges_distribution,
            max_weight,
        )
        spectra_encoder = SpectraEncoderGrowing.init_from_cfg(cfg)

        if (ckpt_path := cfg.spectra_encoder.get("checkpoint_path")) is not None:
            model_dict = torch.load(BASE_DIR / ckpt_path, map_location=device)
            spectra_encoder.load_state_dict(model_dict)
            spectra_encoder.eval()

        arch.spectra_encoder = spectra_encoder
        arch.merge_function = nn.Linear(
            cfg.spectra_encoder.output_size, cfg.dataset.morgan_nbits
        )

        return arch

    @property
    def global_features(self) -> int:
        gf = super().global_features
        if "spectra_fingerprint" in self.extra_features:
            gf += self.morgan_nbits
        return gf

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor,
        num_peaks: torch.Tensor = None,
        types: torch.Tensor = None,
        instruments: torch.Tensor = None,
        ion_vec: torch.Tensor = None,
        form_vec: torch.Tensor = None,
        intens: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run the forward pass for Spec2MolMetaArch.
        This provides explicit arguments for spectral features while leveraging
        DigressMetaArch's noise logic.
        """
        return super().forward(
            nodes,
            edges,
            mask,
            t,
            num_peaks,
            types,
            instruments,
            ion_vec,
            form_vec,
            intens,
        )

    def forward_backward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        mask: torch.Tensor,
        num_peaks: torch.Tensor = None,
        types: torch.Tensor = None,
        instruments: torch.Tensor = None,
        ion_vec: torch.Tensor = None,
        form_vec: torch.Tensor = None,
        intens: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Noise the input batch, predict the clean graph, compute loss and backpropagate.

        Wraps DigressMetaArch.forward_backward with explicit spectral arguments
        that are forwarded to compute_extra_features for spectra fingerprinting.

        Args:
            nodes: Node features of shape (bs, n, natoms).
            edges: Edge features of shape (bs, n, n, nedges).
            mask: Mask tensor of shape (bs, n).
            num_peaks: Number of peaks per spectrum.
            types: Peak type tensors.
            instruments: Instrument identifiers.
            ion_vec: Ion vector representation.
            form_vec: Formula vector representation.
            intens: Peak intensity tensors.

        Returns:
            tuple: (pN, pE, loss) — predicted node/edge probabilities and scalar loss.
                All may be None on non-last ranks in pipeline parallelism.
        """
        return super().forward_backward(
            nodes,
            edges,
            mask,
            num_peaks,
            types,
            instruments,
            ion_vec,
            form_vec,
            intens,
        )

    def compute_extra_features(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor,
        num_peaks,
        types,
        instruments,
        ion_vec,
        form_vec,
        intens,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute extra node and global features for the current graph state.

        Delegates standard features (laplacian, node_cycle, valence) to the parent,
        then appends the spectra fingerprint if configured.
        """
        # Temporarily hide spectra_fingerprint so the parent doesn't hit ValueError
        all_features = self.extra_features
        self.extra_features = [
            f for f in all_features if f in DigressMetaArch.known_extra_features
        ]

        pos_emb, y = super().compute_extra_features(nodes, edges, mask, t)

        # Restore full feature list
        self.extra_features = all_features

        if "spectra_fingerprint" in all_features:
            fp = spectra_fingerprint(
                num_peaks,
                types,
                instruments,
                ion_vec,
                form_vec,
                intens,
                self.spectra_encoder,
                self.merge_function,
            )
            y = torch.cat([y, fp], dim=-1)

        return pos_emb, y

    def trainable_parameters(self):
        """Yield from the trainable parameters of the model."""
        yield from super().trainable_parameters()
        yield from self.merge_function.parameters()

    def checkpoint_state_dict(self) -> dict:
        """Return the state dict for checkpointing."""
        state = super().checkpoint_state_dict()
        state["merge_function"] = self.merge_function.state_dict()
        return state

    def load_checkpoint_state_dict(self, state: dict):
        """Load the model state from a checkpoint state dict."""
        super().load_checkpoint_state_dict(state)
        if "merge_function" in state:
            self.merge_function.load_state_dict(state["merge_function"])

    def sync_extra_gradients(self):
        """Synchronize merge_function gradients across ranks in DDP."""
        if torch.distributed.is_initialized():
            for param in self.merge_function.parameters():
                if param.grad is not None:
                    torch.distributed.all_reduce(
                        param.grad.data, op=torch.distributed.ReduceOp.AVG
                    )
