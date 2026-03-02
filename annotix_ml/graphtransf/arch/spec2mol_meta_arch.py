from typing import Literal
import torch
from omegaconf import DictConfig

from annotix_ml.graphtransf.arch import DigressMetaArch


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
        )

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

        return arch

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor,
        fp: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run the forward pass for Spec2MolMetaArch.
        This provides explicit arguments for spectral features while leveraging
        DigressMetaArch's noise logic.
        """
        return super().forward(nodes, edges, mask, t, fp)

    def forward_backward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        mask: torch.Tensor,
        fp: torch.Tensor,
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
        return super().forward_backward(nodes, edges, mask, fp)

    def trainable_parameters(self):
        """Yield from the trainable parameters of the model."""
        yield from super().trainable_parameters()

    def checkpoint_state_dict(self) -> dict:
        """Return the state dict for checkpointing."""
        state = super().checkpoint_state_dict()
        return state

    def load_checkpoint_state_dict(self, state: dict):
        """Load the model state from a checkpoint state dict."""
        super().load_checkpoint_state_dict(state)
