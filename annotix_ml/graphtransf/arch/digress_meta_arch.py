import os
from typing import Literal

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.linalg import LinAlgError
from tqdm import tqdm

from annotix_ml.graphtransf.data.atoms_data import TYPE_EDGES, VALID_ELEMENTS
from annotix_ml.graphtransf.data.data_utils import mask_any_tensor
from annotix_ml.graphtransf.models.gnn import GnnNodeEdges, GnnNodeEdgesWithoutY
from annotix_ml.graphtransf.models.noising import NoisingModel
from annotix_ml.graphtransf.math.extra_features import (
    laplacian_embedding,
    node_cycle,
    valency,
    charge,
    weight,
)
from annotix_ml.graphtransf.math.metrics import compute_accuracy
from annotix_ml.graphtransf.math.noising import sample_discrete_features


class DigressMetaArch:
    """
    Architecture for the DiGress model.

    This class handles the initialization of the diffusion model (GNN) and the noising model,
    as well as the computation of extra features and the forward pass for training and generation.

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
        no_y (bool, optional): If True, use a model without global feature updates (`GnnNodeEdgesWithoutY`). Defaults to False.
        nodes_distribution (torch.Tensor | None, optional): Marginal distribution of node types. Required if noise_strategy is "distribution".
        edges_distribution (torch.Tensor | None, optional): Marginal distribution of edge types. Required if noise_strategy is "distribution".
        k (int | None, optional): Number of eigenvectors for Laplacian embedding. Required if "laplacian_embedding" is in `extra_features`.
        extra_features (list[str] | None, optional): List of names of extra features to compute.
        last_layer (Literal["mlp", "unembedding"], optional): Type of the last layer in the GNN. Defaults to "mlp".
        max_weight (float | None, optional): Maximum molecular weight for normalization. Required if "valence_features" is in `extra_features`.
    """

    known_extra_features = ["laplacian_embedding", "node_cycle", "valence_features"]

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
    ):
        # Store information
        self.loss_ratio = loss_ratio
        self.device = device

        # Store extra features information
        self.extra_features = extra_features
        self.num_ev = k
        self.max_weight = max_weight

        for name in extra_features:
            if name not in self.known_extra_features:
                raise ValueError(f"{name} is not a known extra feature.")

        # Instanciate the diffusion model
        self.valid_elements = valid_elements
        self.natoms = len(valid_elements)
        self.nbonds = len(TYPE_EDGES)

        if no_y:
            self.diffuser = GnnNodeEdgesWithoutY(
                d=d,
                de=de,
                n_heads=n_heads,
                node_features=self.node_features,
                global_features=self.global_features,
                n_layers=n_layers,
                natoms=self.natoms,
                nbonds=self.nbonds,
                last_layer=last_layer,
            ).to(device)

        else:
            self.diffuser = GnnNodeEdges(
                d=d,
                de=de,
                dy=dy,
                n_heads=n_heads,
                y_update=y_update,
                node_features=self.node_features,
                global_features=self.global_features,
                n_layers=n_layers,
                natoms=self.natoms,
                nbonds=self.nbonds,
                last_layer=last_layer,
            ).to(device)

        self.no_y = no_y

        # Instanciate the noising model
        if noise_strategy == "uniform":
            nodes_distribution = torch.ones(self.natoms) / self.natoms
            edges_distribution = torch.ones(len(TYPE_EDGES)) / len(TYPE_EDGES)

        elif noise_strategy == "distribution":
            pass

        else:
            raise ValueError(f"{noise_strategy} is not a recognized noise strategy.")

        if (nodes_distribution is None) | (edges_distribution is None):
            raise TypeError("edges or nodes distribution not given.")

        self.nodes_distribution = nodes_distribution
        self.edges_distribution = edges_distribution

        self.noiser = NoisingModel(
            diffusion_steps=diffusion_steps,
            nodes_distribution=nodes_distribution,
            edges_distribution=edges_distribution,
        )
        self.noiser.move_to(device)

        # Make the loss
        self.loss = nn.CrossEntropyLoss()

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
        Initialize the DigressMetaArch model from a configuration object.
        Refer to `configs/default_config.yaml` for the required configuration keys.

        Args:
            cfg (DictConfig): Configuration object containing model and training parameters.
                Expected keys include `model.d`, `model.de`, `model.dy`, `model.n_heads`, etc.
            device (torch.device): Device on which the model should be initialized.
            valid_elements (list[str] | None): List of valid atomic element symbols.
            nodes_distribution (torch.Tensor | None): Marginal distribution of node types.
            edges_distribution (torch.Tensor | None): Marginal distribution of edge types.
            max_weight (float | None, optional): Maximum molecular weight for normalization. Defaults to None.

        Returns:
            DigressMetaArch: Initialized model instance.
        """
        if not valid_elements:
            valid_elements = list(VALID_ELEMENTS)

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
            last_layer=cfg.model.last_layer,
            valid_elements=valid_elements,
            nodes_distribution=nodes_distribution,
            edges_distribution=edges_distribution,
            max_weight=max_weight,
        )

    @classmethod
    def load_pretrained(
        cls,
        cfg: DictConfig,
        device: torch.device,
        model_path: str,
        valid_elements: list[str] | None,
        nodes_distribution: torch.Tensor | None,
        edges_distribution: torch.Tensor | None,
        max_weight: float | None = None,
    ):
        """
        Load a pretrained DigressMetaArch model from a file.
        Note that only the weights for the diffuser model are loaded from the specified file.

        Args:
            cfg (DictConfig): Configuration object. Refer to `configs/default_config.yaml`
                for the necessary configuration keys.
            device (torch.device): Device on which the model should be loaded.
            model_path (str): Path to the saved weights file (containing the diffuser state_dict).
            valid_elements (list[str] | None): List of valid atomic element symbols.
            nodes_distribution (torch.Tensor | None): Marginal distribution of node types.
            edges_distribution (torch.Tensor | None): Marginal distribution of edge types.
            max_weight (float | None, optional): Maximum molecular weight for normalization. Defaults to None.

        Returns:
            DigressMetaArch: Loaded model instance.
        """
        if not os.path.exists(model_path):
            raise ValueError(f"{model_path} is not an existing path.")

        model = cls.init_from_cfg(
            cfg,
            device,
            valid_elements,
            nodes_distribution,
            edges_distribution,
            max_weight,
        )

        saved_model = torch.load(model_path, weights_only=True)
        model.diffuser.load_state_dict(saved_model)

        return model

    @property
    def global_features(self) -> int:
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

        return global_features

    @property
    def node_features(self) -> int:
        node_features = 0
        for name in self.extra_features:
            if name == "laplacian_embedding":
                node_features += self.num_ev + 1  # eigvalues + num_connected_components

            elif name == "node_cycle":
                node_features += 3

            elif name == "valence_features":
                node_features += 2

        return node_features

    def compute_extra_features(self, batch: dict):
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

    def forward(self, batch):
        """
        Run the forward pass: noise the input batch, then predict the clean graph.
        This method first adds noise to the input nodes and edges using `self.noiser`,
        and then uses `self.diffuser` to predict the clean graph probabilities.

        Args:
            batch (dict[str, torch.Tensor]): A dictionary containing:
                - "nodes" (torch.Tensor): Node features of shape (bs, n, natoms).
                - "edges" (torch.Tensor): Edge features of shape (bs, n, n, nedges).
                - "mask" (torch.Tensor): Mask tensor of shape (bs, n).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - pN (torch.Tensor): Predicted node probabilities of shape (bs, n, natoms).
                - pE (torch.Tensor): Predicted edge probabilities of shape (bs, n, n, nedges).
        """
        # unpack the elements of the batch
        N = batch["nodes"]
        E = batch["edges"]
        mask = batch["mask"]

        # noise the graph
        N_noised, E_noised, sampled_t = self.noiser(N, E, mask)

        # update the batch with noised elements
        batch["nodes"] = N_noised
        batch["edges"] = E_noised
        batch["t"] = sampled_t

        # compute the extra features
        pos_emb, y = self.compute_extra_features(
            batch
        )  # (bs, node_features), (bs, global_features)

        # Compute the output of the diffuser
        pN, pE, _ = self.diffuser(
            batch["nodes"], batch["edges"], pos_emb, y, mask
        )  # (bs, n, n_atoms), (bs, n, n, n_edges)

        return pN, pE

    def forward_backward(self, batch):
        """
        Compute the forward pass followed by the loss and metrics computation.

        Args:
            batch (dict[str, torch.Tensor]): A dictionary containing:
                - "nodes" (torch.Tensor): Node features of shape (bs, n, natoms).
                - "edges" (torch.Tensor): Edge features of shape (bs, n, n, nedges).
                - "mask" (torch.Tensor): Mask tensor of shape (bs, n).

        Returns:
            tuple[torch.Tensor, dict]: A tuple containing:
                - total_loss (torch.Tensor): Scalar loss value.
                - metrics (dict): Dictionary of computed metrics (accuracy, cross-entropy per atom).
        """
        # unpack the elements of the batch
        N = batch["nodes"]
        E = batch["edges"]
        mask = batch["mask"]

        # Compute the output of the diffuser
        pN, pE = self.forward(batch)  # (bs, n, n_atoms), (bs, n, n, n_edges)

        # Compute the losses
        # Node loss
        N_target = N.argmax(-1)  # (bs, n)
        N_target = mask_any_tensor(
            N_target, mask, fill=-100
        )  # -100 is the ignore_index of CrossEntropLoss
        pN = pN.transpose(1, 2)  # (bs, n_atoms, n)

        Nloss = self.loss(pN, N_target)

        # Edges loss
        E_target = E.argmax(-1)  # (bs, n, n)
        E_target = mask_any_tensor(
            E_target, mask, fill=-100
        )  # -100 is the ignore_index of CrossEntropLoss
        pE = pE.permute((0, 3, 1, 2))  # (bs, n_edges, n, n)

        Eloss = self.loss(pE, E_target)

        # Add the losses
        total_loss = Nloss + (self.loss_ratio * Eloss)

        # Compute the accuracy
        accuracy = compute_accuracy(
            pN.transpose(1, 2), pE.permute((0, 2, 3, 1)), N, E, mask
        )

        # Compute the cross-entropy for each of the atoms
        metrics = {}
        for idx, at in enumerate(self.valid_elements):
            input = pN.transpose(1, 2).softmax(-1)[..., idx]  # (bs, n, 1)
            target = N[..., idx]
            mask_bool = mask.bool()

            # input: (bs, n, 1) -> (bs, n)
            input_masked = input.squeeze(-1)[mask_bool]
            # target: (bs, n)
            target_masked = target[mask_bool].float()

            ce = nn.functional.binary_cross_entropy(input_masked, target_masked)
            metrics[f"ce_{at}"] = ce

        metrics |= {
            "node_accuracy": torch.tensor(accuracy["node_accuracy"]).mean().item(),
            "edge_accuracy": torch.tensor(accuracy["edge_accuracy"]).mean().item(),
        }

        return total_loss, metrics

    @torch.no_grad()
    def generate(
        self,
        num_samples: int | torch.Tensor,
        max_nodes: int,
        min_nodes: int = 1,
        progress_bar=False,
        num_attempts: int = 3,
    ):
        """
        Generate new graphs using the reverse diffusion process.

        The generation starts from pure noise (sampled according to the marginal distributions)
        and iteratively applies the reverse diffusion step. At each step `t`, the model
        predicts the clean graph, and the sample for `t-1` is drawn from the posterior
        distribution `q(z_{t-1} | z_t, z_0)`.

        Args:
            num_samples (int | torch.Tensor):
                - If int: Number of samples to generate. The size (node count) of each
                  generated graph is randomly sampled between `min_nodes` and `max_nodes`.
                - If torch.Tensor: A 1D tensor of shape (num_samples,) where each element
                  specifies the exact number of nodes for a sample. The total number of
                  samples is then determined by the length of this tensor.
            max_nodes (int): Maximum number of nodes for the generated graphs.
            min_nodes (int, optional): Minimum number of nodes for the generated graphs. Defaults to 1.
            progress_bar (bool, optional): Whether to show a progress bar during denoising. Defaults to False.
            num_attempts (int, optional): Number of attempts to generate graphs (to handle potential LinAlgError). Defaults to 3.

        Raises:
            LinAlgError: If generation fails after all attempts due to numerical instability.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - N (torch.Tensor): Generated node features of shape (bs, n, natoms).
                - E (torch.Tensor): Generated edge features of shape (bs, n, n, nedges).
                - mask (torch.Tensor): Generated masks of shape (bs, n).
        """
        i = 0
        while i < num_attempts:
            try:
                # sample random n
                if isinstance(num_samples, int):
                    n = torch.randint(min_nodes, max_nodes, (num_samples,))

                elif isinstance(num_samples, torch.Tensor):
                    n = num_samples
                    assert len(num_samples.shape) == 1, (
                        f"Wrong shape for num_samples: {num_samples.shape}"
                    )
                    num_samples = n.shape[0]

                # Make the mask
                mask = torch.stack(
                    [
                        torch.cat([torch.ones(int(m)), torch.zeros(max_nodes - int(m))])
                        for m in n
                    ]
                )  # (bs, max_nodes)
                mask = mask.to(self.device)

                # Make the distributions based on the dataset
                N_dist = (
                    self.nodes_distribution.unsqueeze(0)
                    .unsqueeze(0)
                    .expand(num_samples, max_nodes, -1)
                )  # (bs, n, n_atoms)
                E_dist = (
                    self.edges_distribution.unsqueeze(0)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .expand(num_samples, max_nodes, max_nodes, -1)
                )  # (bs, n, n, n_edges)

                # sample a random graph
                N, E = sample_discrete_features(
                    N_dist, E_dist, mask
                )  # (bs, n, n_atoms), (bs, n, n, n_edges)

                # Denoise the graph
                self.diffuser.eval()

                if progress_bar:
                    t_range = tqdm(
                        reversed(range(0, self.noiser.T)),
                        total=self.noiser.T,
                        desc="Denoising",
                    )
                else:
                    t_range = reversed(range(0, self.noiser.T))

                for t in t_range:
                    # Convert to float for compatibility with noising model
                    N = N.float().to(self.device)
                    E = E.float().to(self.device)

                    # Mask the graph
                    N = mask_any_tensor(N, mask, fill=0)  # (bs, n, n_atoms)
                    E = mask_any_tensor(E, mask, fill=0)  # (bs, n, n, n_edges)

                    t_tensor = (
                        torch.tensor(t).unsqueeze(-1).expand((num_samples, -1))
                    )  # bs, 1

                    # Create a temporary batch for compute_extra_features
                    temp_batch = {
                        "nodes": N,
                        "edges": E,
                        "mask": mask,
                        "t": t_tensor,
                    }
                    pos_emb, y = self.compute_extra_features(temp_batch)

                    pN, pE, _ = self.diffuser(
                        N, E, pos_emb, y, mask
                    )  # (bs, n, n_atoms), (bs, n, n, n_edges)

                    pN = pN.softmax(-1)  #  (bs, n, n_atoms)
                    pE = pE.softmax(-1)  #  (bs, n, n, n_edges)

                    # Compute the posterior distribution
                    post_N, post_E = self.noiser.get_posterior(
                        N, E, t
                    )  # (bs, n, n_atoms, n_atoms), (bs, n, n, n_edges, n_edges)

                    # Compute the combined distribution -> element-wise multiplication
                    # Then sum over all the possible starting values (dim -2)
                    probN = (post_N * pN.unsqueeze(-1)).sum(dim=-2)  # (bs, n, n_atoms)
                    probE = (post_E * pE.unsqueeze(-1)).sum(
                        dim=-2
                    )  # (bs, n, n, n_edges)

                    # Normalize with epsilon to prevent division by zero
                    eps = 1e-6
                    probN = probN / (
                        probN.sum(dim=-1, keepdim=True) + eps
                    )  # (bs, n, n_atoms)
                    probE = probE / (
                        probE.sum(dim=-1, keepdim=True) + eps
                    )  # (bs, n, n, n_edges)

                    # Sample from the combined distribution
                    N, E = sample_discrete_features(
                        probN, probE, mask
                    )  # (bs, n, n_atoms), (bs, n, n, n_edges)

                return N, E, mask

            except LinAlgError:
                print(f"Generation failed, attempt {i + 1} / {num_attempts}")
                i += 1

        raise LinAlgError("Impossible to generate graphs with current model.")
