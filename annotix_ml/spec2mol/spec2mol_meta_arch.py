from typing import Literal
import torch
import torch.nn as nn
from omegaconf import DictConfig

from annotix_ml.graphtransf.arch.digress_meta_arch import DigressMetaArch
from annotix_ml.spectraencoder.model.spectra_encoder import SpectraEncoder
from annotix_ml.spec2mol.extra_features import spectra_fingerprint


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
        hidden_size: int = 50,
        spectra_dropout: float = 0.0,
        top_layers: int = 1,
        magma_modulo: int = 2048,
        peak_attn_layers: int = 2,
        set_pooling: str = "intensity",
        pairwise_featurization: bool = False,
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
        self.spectra_encoder = SpectraEncoder(
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
        return cls(
            d=cfg.model.d,
            de=cfg.model.de,
            dy=cfg.model.dy,
            n_heads=cfg.model.n_heads,
            n_layers=cfg.model.n_layers,
            noise_strategy=cfg.model.noise_strategy,
            diffusion_steps=cfg.model.diffusion_steps,
            loss_ratio=cfg.train.loss_ratio,
            device=device,
            valid_elements=valid_elements,
            y_update=cfg.model.y_update,
            no_y=cfg.model.no_y,
            nodes_distribution=nodes_distribution,
            edges_distribution=edges_distribution,
            k=cfg.model.num_ev,
            extra_features=list(cfg.model.extra_features),
            max_weight=max_weight,
            morgan_nbits=cfg.dataset.morgan_nbits,
            # SpectraEncoder args
            form_embedder=cfg.spectra_encoder.form_embedder,
            output_size=cfg.spectra_encoder.output_size,
            hidden_size=cfg.spectra_encoder.hidden_size,
            spectra_dropout=cfg.spectra_encoder.spectra_dropout,
            top_layers=cfg.spectra_encoder.top_layers,
            magma_modulo=cfg.spectra_encoder.magma_modulo,
            peak_attn_layers=cfg.spectra_encoder.peak_attn_layers,
            set_pooling=cfg.spectra_encoder.set_pooling,
            pairwise_featurization=cfg.spectra_encoder.pairwise_featurization,
            num_heads=cfg.spectra_encoder.num_heads,
            embed_instrument=cfg.spectra_encoder.embed_instrument,
            inten_transform=cfg.spectra_encoder.inten_transform,
            no_diffs=cfg.spectra_encoder.no_diffs,
        )

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
            nodes=nodes,
            edges=edges,
            mask=mask,
            t=t,
            num_peaks=num_peaks,
            types=types,
            instruments=instruments,
            ion_vec=ion_vec,
            form_vec=form_vec,
            intens=intens,
        )

    def compute_extra_features(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
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
                kwargs["num_peaks"],
                kwargs["types"],
                kwargs["instruments"],
                kwargs["ion_vec"],
                kwargs["form_vec"],
                kwargs["intens"],
                self.spectra_encoder,
                self.merge_function,
            )
            y = torch.cat([y, fp], dim=-1)

        return pos_emb, y
