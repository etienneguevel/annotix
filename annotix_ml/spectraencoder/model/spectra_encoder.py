from typing import Tuple

import torch
from torch import nn

from annotix_ml.spectraencoder.model.modules import FormulaTransformer, FPGrowingModule


from omegaconf import DictConfig


class SpectraEncoder(nn.Module):
    """SpectraEncoder."""

    def __init__(
        self,
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
        super(SpectraEncoder, self).__init__()

        spectra_encoder_main = FormulaTransformer(
            hidden_size=hidden_size,
            peak_attn_layers=peak_attn_layers,
            set_pooling=set_pooling,
            spectra_dropout=spectra_dropout,
            pairwise_featurization=pairwise_featurization,
            num_heads=num_heads,
            output_size=output_size,
            form_embedder=form_embedder,
            embed_instrument=embed_instrument,
            inten_transform=inten_transform,
            no_diffs=no_diffs,
        )

        fragment_pred_parts = []
        for _ in range(top_layers - 1):
            fragment_pred_parts.append(nn.Linear(hidden_size, hidden_size))
            fragment_pred_parts.append(nn.ReLU())
            fragment_pred_parts.append(nn.Dropout(spectra_dropout))

        fragment_pred_parts.append(nn.Linear(hidden_size, magma_modulo))

        fragment_predictor = nn.Sequential(*fragment_pred_parts)

        top_layer_parts = []
        for _ in range(top_layers - 1):
            top_layer_parts.append(nn.Linear(hidden_size, hidden_size))
            top_layer_parts.append(nn.ReLU())
            top_layer_parts.append(nn.Dropout(spectra_dropout))
        top_layer_parts.append(nn.Linear(hidden_size, output_size))
        top_layer_parts.append(nn.Sigmoid())
        spectra_predictor = nn.Sequential(*top_layer_parts)

        self.spectra_encoder = nn.ModuleList(
            [spectra_encoder_main, fragment_predictor, spectra_predictor]
        )

    @classmethod
    def init_from_cfg(cls, cfg: DictConfig):
        """
        Initialize the SpectraEncoder model from a configuration object.

        Args:
            cfg (DictConfig): Configuration object containing 'spectra_encoder' parameters.

        Returns:
            SpectraEncoder: Initialized model instance.
        """
        return cls(
            form_embedder=cfg.spectra_encoder.form_embedder,
            output_size=cfg.spectra_encoder.output_size,
            hidden_size=cfg.spectra_encoder.hidden_size,
            spectra_dropout=cfg.spectra_encoder.spectra_dropout,
            top_layers=cfg.spectra_encoder.top_layers,
            magma_modulo=cfg.spectra_encoder.magma_modulo,
            peak_attn_layers=cfg.spectra_encoder.peak_attn_layers,
            set_pooling=cfg.spectra_encoder.get("set_pooling", "intensity"),
            pairwise_featurization=cfg.spectra_encoder.pairwise_featurization,
            num_heads=cfg.spectra_encoder.num_heads,
            embed_instrument=cfg.spectra_encoder.embed_instrument,
            inten_transform=cfg.spectra_encoder.inten_transform,
            no_diffs=cfg.spectra_encoder.no_diffs,
        )

    def forward(self, batch: dict) -> Tuple[torch.Tensor, dict]:
        """Forward pass."""
        encoder_output, aux_out = self.spectra_encoder[0](batch, return_aux=True)

        pred_frag_fps = self.spectra_encoder[1](aux_out["peak_tensor"])
        aux_outputs = {"pred_frag_fps": pred_frag_fps}

        output = self.spectra_encoder[2](encoder_output)
        aux_outputs["h0"] = encoder_output

        return output, aux_outputs


class SpectraEncoderGrowing(nn.Module):
    """SpectraEncoder."""

    def __init__(
        self,
        form_embedder: str = "float",
        output_size: int = 4096,
        hidden_size: int = 50,
        spectra_dropout: float = 0.0,
        top_layers: int = 1,
        refine_layers: int = 0,
        magma_modulo: int = 2048,
        peak_attn_layers: int = 2,
        set_pooling: str = "intensity",
        pairwise_featurization: bool = False,
        num_heads: int = 8,
        embed_instrument: bool = False,
        inten_transform: str = "float",
        no_diffs: bool = False,
    ):
        super(SpectraEncoderGrowing, self).__init__()

        spectra_encoder_main = FormulaTransformer(
            hidden_size=hidden_size,
            peak_attn_layers=peak_attn_layers,
            set_pooling=set_pooling,
            spectra_dropout=spectra_dropout,
            pairwise_featurization=pairwise_featurization,
            num_heads=num_heads,
            output_size=output_size,
            form_embedder=form_embedder,
            embed_instrument=embed_instrument,
            inten_transform=inten_transform,
            no_diffs=no_diffs,
        )

        fragment_pred_parts = []
        for _ in range(top_layers - 1):
            fragment_pred_parts.append(nn.Linear(hidden_size, hidden_size))
            fragment_pred_parts.append(nn.ReLU())
            fragment_pred_parts.append(nn.Dropout(spectra_dropout))

        fragment_pred_parts.append(nn.Linear(hidden_size, magma_modulo))

        fragment_predictor = nn.Sequential(*fragment_pred_parts)

        spectra_predictor = FPGrowingModule(
            hidden_input_dim=hidden_size,
            final_target_dim=output_size,
            num_splits=refine_layers,
            reduce_factor=2,
        )

        self.spectra_encoder = nn.ModuleList(
            [spectra_encoder_main, fragment_predictor, spectra_predictor]
        )

    @classmethod
    def init_from_cfg(cls, cfg: DictConfig):
        """
        Initialize the SpectraEncoderGrowing model from a configuration object.

        Args:
            cfg (DictConfig): Configuration object containing 'spectra_encoder' parameters.

        Returns:
            SpectraEncoderGrowing: Initialized model instance.
        """
        return cls(
            form_embedder=cfg.spectra_encoder.form_embedder,
            output_size=cfg.spectra_encoder.output_size,
            hidden_size=cfg.spectra_encoder.hidden_size,
            spectra_dropout=cfg.spectra_encoder.spectra_dropout,
            top_layers=cfg.spectra_encoder.top_layers,
            refine_layers=cfg.spectra_encoder.get("refine_layers", 0),
            magma_modulo=cfg.spectra_encoder.magma_modulo,
            peak_attn_layers=cfg.spectra_encoder.peak_attn_layers,
            set_pooling=cfg.spectra_encoder.get("set_pooling", "intensity"),
            pairwise_featurization=cfg.spectra_encoder.pairwise_featurization,
            num_heads=cfg.spectra_encoder.num_heads,
            embed_instrument=cfg.spectra_encoder.embed_instrument,
            inten_transform=cfg.spectra_encoder.inten_transform,
            no_diffs=cfg.spectra_encoder.no_diffs,
        )

    def forward(self, batch: dict) -> Tuple[torch.Tensor, dict]:
        """Forward pass."""
        encoder_output, aux_out = self.spectra_encoder[0](batch, return_aux=True)
        pred_frag_fps = self.spectra_encoder[1](aux_out["peak_tensor"])
        aux_outputs = {"pred_frag_fps": pred_frag_fps}

        output = self.spectra_encoder[2](encoder_output)
        intermediates = output[:-1]
        final_output = output[-1]
        aux_outputs["int_preds"] = intermediates
        output = final_output
        aux_outputs["h0"] = encoder_output

        return output, aux_outputs  # aux_outputs["int_preds"][-1]
