from typing import Any, Literal, Mapping
import torch
import torch.nn as nn
from omegaconf import DictConfig

from annotix_ml.graphtransf.arch.digress_meta_arch import DigressMetaArch
from annotix_ml.graphtransf.math.extra_features import (
    laplacian_embedding,
    node_cycle,
    valency,
    charge,
    weight,
)
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
        last_layer: Literal["mlp", "unembedding"] = "mlp",
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
        """
        Initialize the Spec2MolMetaArch model.

        Args:
            d (int): Hidden dimension for node features.
            de (int): Hidden dimension for edge features.
            dy (int): Hidden dimension for global features.
            n_heads (int): Number of attention heads.
            n_layers (int): Number of GNN layers.
            noise_strategy (Literal["uniform", "distribution"]): Strategy for the noise distribution.
            diffusion_steps (int): Total number of diffusion steps.
            loss_ratio (float): Weight of the edge loss in the total loss.
            device (torch.device): Device on which the model is initialized.
            valid_elements (list[str]): List of valid atomic element symbols.
            y_update (bool, optional): Whether to update global features in the diffusion model. Defaults to True.
            no_y (bool, optional): If True, use a model without global feature updates. Defaults to False.
            nodes_distribution (torch.Tensor | None, optional): Marginal distribution of node types. Defaults to None.
            edges_distribution (torch.Tensor | None, optional): Marginal distribution of edge types. Defaults to None.
            k (int | None, optional): Number of eigenvectors for Laplacian embedding. Defaults to None.
            extra_features (list[str] | None, optional): List of names of extra features to compute. Defaults to None.
            last_layer (Literal["mlp", "unembedding"], optional): Type of the last layer in the GNN. Defaults to "mlp".
            max_weight (float | None, optional): Maximum molecular weight for normalization. Defaults to None.
            morgan_nbits (int, optional): Size of the Morgan fingerprint (used in merging). Defaults to 2048.
            form_embedder (str, optional): Type of formula embedder for SpectraEncoder. Defaults to "float".
            output_size (int, optional): Output size of the SpectraEncoder. Defaults to 4096.
            hidden_size (int, optional): Hidden size for SpectraEncoder layers. Defaults to 50.
            spectra_dropout (float, optional): Dropout rate for SpectraEncoder. Defaults to 0.0.
            top_layers (int, optional): Number of top layers in SpectraEncoder. Defaults to 1.
            magma_modulo (int, optional): Modulo for MAGMA-style embedding. Defaults to 2048.
            peak_attn_layers (int, optional): Number of attention layers for peaks. Defaults to 2.
            set_pooling (str, optional): Pooling strategy for peak features. Defaults to "intensity".
            pairwise_featurization (bool, optional): Whether to use pairwise peak featurization. Defaults to False.
            num_heads (int, optional): Number of heads in SpectraEncoder attention. Defaults to 8.
            embed_instrument (bool, optional): Whether to embed instrument information. Defaults to False.
            inten_transform (str, optional): Transformation for peak intensities. Defaults to "float".
            no_diffs (bool, optional): If True, do not use mass differences in SpectraEncoder. Defaults to False.
        """
        # Store info specific to spectra
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
            last_layer=last_layer,
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
        """
        Initialize the Spec2MolMetaArch model from a configuration object.

        Args:
            cfg (DictConfig): Configuration object containing model and training parameters.
            device (torch.device): Device on which the model should be initialized.
            valid_elements (list[str] | None): List of valid atomic element symbols.
            nodes_distribution (torch.Tensor | None): Marginal distribution of node types.
            edges_distribution (torch.Tensor | None): Marginal distribution of edge types.
            max_weight (float | None, optional): Maximum molecular weight for normalization. Defaults to None.

        Returns:
            Spec2MolMetaArch: Initialized model instance.
        """
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
            last_layer=cfg.model.last_layer,
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
        """
        Total number of global features, including the noising step and extra features.

        Returns:
            int: The dimension of the global features tensor.
        """
        # noising step is always a parameter
        global_features = 1
        for name in self.extra_features:
            if name == "laplacian_embedding":
                global_features += (
                    self.num_ev + 1
                )  # eigvalues + num_connected_components

            elif name == "node_cycle":
                global_features += 4

            elif name == "valence_features":
                global_features += 1

            elif name == "spectra_fingerprint":
                global_features += self.morgan_nbits

        return global_features

    @property
    def node_features(self) -> int:
        """
        Total number of extra node features.

        Returns:
            int: The dimension of the extra node features tensor.
        """
        node_features = 0
        for name in self.extra_features:
            if name == "laplacian_embedding":
                node_features += self.num_ev + 1  # eigvalues + num_connected_components

            elif name == "node_cycle":
                node_features += 3

            elif name == "valence_features":
                node_features += 2

        return node_features

    def compute_extra_features(self, batch: Mapping[str, Any]):
        """
        Compute extra node and global features for the current graph state.

        Args:
            batch (dict): Batch dictionary containing:
                - "nodes" (torch.Tensor): Node features tensor of shape (bs, n, natoms).
                - "edges" (torch.Tensor): Edge features tensor of shape (bs, n, n, nedges).
                - "mask" (torch.Tensor): Mask tensor of shape (bs, n).
                - "t" (torch.Tensor): Timestep tensor of shape (bs, 1).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - node_features (torch.Tensor): Computed node features of shape (bs, n, node_features).
                - global_features (torch.Tensor): Computed global features of shape (bs, global_features).
        """
        # Init the list of node and global_features
        node_features_list = []
        global_features_list = []

        for name in self.extra_features:
            if name == "laplacian_embedding":
                if self.num_ev is None:
                    raise ValueError("k must be specified for laplacian_embedding.")

                node_features, global_features = laplacian_embedding(
                    batch["edges"], self.num_ev, batch["mask"]
                )

            elif name == "node_cycle":
                node_features, global_features = node_cycle(
                    batch["edges"], batch["mask"]
                )

            elif name == "valence_features":
                node_features_val = valency(batch["edges"], batch["mask"])

                node_features_charge = charge(
                    batch["nodes"], batch["edges"], batch["mask"], self.valid_elements
                )

                global_features_weight = weight(
                    batch["nodes"],
                    self.valid_elements,
                    self.max_weight,
                )

                node_features = torch.cat(
                    [node_features_val, node_features_charge], dim=-1
                )
                global_features = global_features_weight

            elif name == "spectra_fingerprint":
                node_features = None
                global_features = spectra_fingerprint(
                    batch["num_peaks"],
                    batch["types"],
                    batch["instruments"],
                    batch["ion_vec"],
                    batch["form_vec"],
                    batch["intens"],
                    self.spectra_encoder,
                    self.merge_function,
                )

            else:
                raise ValueError(f"{name} is not a recognized extra feature.")

            # Unpack for cases with global and local outputs
            if node_features is not None:
                node_features_list.append(node_features)

            if global_features is not None:
                global_features_list.append(global_features)

            global_features = None
            node_features = None

        pos_emb = torch.cat(node_features_list, dim=-1)  # (bs, n, n_features)
        y = torch.cat(global_features_list, dim=-1)  # (bs, n_global_features)

        # Add the noising step to y
        t = batch["t"]
        t = t.to(device=y.device, dtype=y.dtype) / self.noiser.T
        y = torch.cat([y, t], dim=-1)

        return pos_emb, y
