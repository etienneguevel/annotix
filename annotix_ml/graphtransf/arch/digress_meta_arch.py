import os
from functools import partial
from typing import Literal


import torch
import torch.nn as nn
from omegaconf import DictConfig
from tqdm import tqdm

from annotix_ml.graphtransf.data.atoms_data import TYPE_EDGES
from annotix_ml.graphtransf.data.data_utils import mask_any_tensor
from annotix_ml.graphtransf.data.dataset import GraphDatasetFromSMILEs
from annotix_ml.graphtransf.models.gnn import GnnNodeEdges
from annotix_ml.graphtransf.models.noising import NoisingModel
from annotix_ml.graphtransf.math.extra_features import laplacian_embedding, node_cycle
from annotix_ml.graphtransf.math.metrics import compute_accuracy
from annotix_ml.graphtransf.math.noising import sample_discrete_features


class DigressMetaArch:
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
        train_dataset: GraphDatasetFromSMILEs,
        k: int | None = None,
        extra_features: list[str] | None = None,
        last_layer: Literal["mlp", "unembedding"] = "mlp",
    ):
        # Store information
        self.loss_ratio = loss_ratio
        self.device = device

        # Make the extra_features functions
        extra_features_functions = []
        node_features = 0
        global_features = 0

        if extra_features is None:
            extra_features = []

        for name in extra_features:
            if name == "laplacian_embedding":
                if k is None:
                    raise ValueError("k must be specified for laplacian_embedding.")

                f = partial(laplacian_embedding, k=k)
                extra_features_functions.append(f)
                node_features += k
                global_features += k

            elif name == "node_cycle":
                extra_features_functions.append(node_cycle)
                node_features += 3
                global_features += 4

            else:
                raise ValueError(f"{name} is not a recognized extra feature.")

        self.extra_features = extra_features_functions

        # Instanciate the diffusion model
        self.natoms = len(train_dataset.valid_elements)
        self.nbonds = len(TYPE_EDGES)

        self.diffuser = GnnNodeEdges(
            d=d,
            de=de,
            dy=dy,
            n_heads=n_heads,
            node_features=node_features,
            global_features=global_features,
            n_layers=n_layers,
            natoms=self.natoms,
            nbonds=self.nbonds,
            last_layer=last_layer,
        ).to(device)

        # Instanciate the noising model
        if noise_strategy == "uniform":
            nodes_distribution = torch.ones(self.natoms) / self.natoms
            edges_distribution = torch.ones(len(TYPE_EDGES)) / len(TYPE_EDGES)

        elif noise_strategy == "distribution":
            nodes_distribution = train_dataset.nodes_distribution
            edges_distribution = train_dataset.edges_distribution

        else:
            raise ValueError(f"{noise_strategy} is not a recognized noise strategy.")

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
        train_dataset: GraphDatasetFromSMILEs,
    ):
        return cls(
            d=cfg.model.d,
            de=cfg.model.de,
            dy=cfg.model.dy,
            n_heads=cfg.model.n_heads,
            n_layers=cfg.model.n_layers,
            train_dataset=train_dataset,
            noise_strategy=cfg.model.noise_strategy,
            diffusion_steps=cfg.model.diffusion_steps,
            loss_ratio=cfg.train.loss_ratio,
            device=device,
            k=cfg.model.num_ev,
            extra_features=cfg.model.extra_features,
            last_layer=cfg.model.last_layer,
        )

    @classmethod
    def load_pretrained(
        cls,
        cfg: DictConfig,
        device: torch.device,
        train_dataset: GraphDatasetFromSMILEs,
        model_path: str,
    ):
        if not os.path.exists(model_path):
            raise ValueError(f"{model_path} is not an existing path.")

        model = cls.init_from_cfg(cfg, device, train_dataset)

        saved_model = torch.load(model_path, weights_only=True)
        model.diffuser.load_state_dict(saved_model)

        return model

    def compute_extra_features(
        self,
        edges: torch.Tensor,
        mask: torch.Tensor,
    ):
        # get the extra features
        node_features, global_features = list(
            zip(
                *[
                    extra_feature(edges=edges, mask=mask)
                    for extra_feature in self.extra_features
                ]
            )
        )

        if len(node_features) == 1:
            pos_emb = node_features[0]
            y = global_features[0]

        else:
            pos_emb = torch.cat(node_features, dim=-1)  # (bs, n, n_features)
            y = torch.cat(global_features, dim=-1)  # (bs, n_global_features)

        return pos_emb, y

    def forward(self, batch):
        # unpack the elements of the batch
        N, E, mask = batch  # (bs, n, n_atoms), (bs, n, n, n_edges), (bs,)

        # noise the graph
        N_noised, E_noised, _ = self.noiser(N, E, mask)

        # compute the extra features
        pos_emb, y = self.compute_extra_features(
            E_noised, mask
        )  # (bs, node_features), (bs, n)

        # Compute the output of the diffuser
        pN, pE, _ = self.diffuser(
            N_noised, E_noised, pos_emb, y, mask
        )  # (bs, n, n_atoms), (bs, n, n, n_edges)

        return pN, pE

    def forward_backward(self, batch):
        # unpack the elements of the batch
        N, E, mask = batch  # (bs, n, n_atoms), (bs, n, n, n_edges), (bs,)

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
        accuracy = {
            "node_accuracy": torch.tensor(accuracy["node_accuracy"]).mean().item(),
            "edge_accuracy": torch.tensor(accuracy["edge_accuracy"]).mean().item(),
        }

        return total_loss, accuracy

    @torch.no_grad()
    def generate(
        self, batch_size: int, max_nodes: int, min_nodes: int = 1, progress_bar=False
    ):
        # sample random n
        n = torch.randint(min_nodes, max_nodes, (batch_size,))

        # Make the mask
        mask = torch.stack(
            [
                torch.cat([torch.ones(int(m)), torch.zeros(max_nodes - int(m))])
                for m in n
            ]
        )  # (bs, max_nodes)
        mask = mask.to(self.device)

        # Make the distributions based on the noiser
        N_dist = (
            self.noiser.n_m.unsqueeze(0).unsqueeze(0).expand(batch_size, max_nodes, -1)
        )  # (bs, n, n_atoms)
        E_dist = (
            self.noiser.e_m.unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, max_nodes, max_nodes, -1)
        )  # (bs, n, n, n_edges)

        # sample a random graph
        N, E = sample_discrete_features(
            N_dist, E_dist, mask
        )  # (bs, n, n_atoms), (bs, n, n, n_edges)

        # Denoise the graph
        self.diffuser.eval()

        if progress_bar:
            t_range = tqdm(
                reversed(range(0, self.noiser.T)), total=self.noiser.T, desc="Denoising"
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

            # Compute the probabilities computed by the model
            pN, pE = self.forward(
                (N, E, mask)
            )  #  (bs, n, n_atoms), (bs, n, n, n_edges)

            pN = pN.softmax(-1)  #  (bs, n, n_atoms)
            pE = pE.softmax(-1)  #  (bs, n, n, n_atoms)

            # Compute the posterior distribution
            post_N, post_E = self.noiser.get_posterior(
                N, E, t
            )  # (bs, n, n_atoms, n_atoms), (bs, n, n, n_edges, n_edges)

            # Compute the combined distribution -> element-wise multiplication
            # Then sum over all the possible starting values (dim -2)
            probN = (post_N * pN.unsqueeze(-1)).sum(dim=-2)  # (bs, n, n_atoms)
            probE = (post_E * pE.unsqueeze(-1)).sum(dim=-2)  # (bs, n, n, n_edges)

            # Normalize with epsilon to prevent division by zero
            eps = 1e-6
            probN = probN / (probN.sum(dim=-1, keepdim=True) + eps)  # (bs, n, n_atoms)
            probE = probE / (
                probE.sum(dim=-1, keepdim=True) + eps
            )  # (bs, n, n, n_edges)

            # Sample from the combined distribution
            N, E = sample_discrete_features(probN, probE, mask)

        return N, E, mask
