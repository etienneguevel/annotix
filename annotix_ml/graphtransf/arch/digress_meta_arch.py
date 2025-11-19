from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn

from annotix_ml.graphtransf.data.atoms_data import VALID_ELEMENTS, TYPE_EDGES
from annotix_ml.graphtransf.data.data_utils import mask_any_tensor
from annotix_ml.graphtransf.models.gnn import GnnNodeEdges
from annotix_ml.graphtransf.models.noising import NoisingModel
from annotix_ml.graphtransf.math.extra_features import laplacian_embedding, node_cycle


class DigressMetaArch:
    def __init__(
        self,
        d: int,
        de: int,
        dy: int,
        n_heads: int,
        n_layers: int,
        nodes_distribution: list[float],
        edges_distribution: list[float],
        diffusion_steps: int,
        loss_ratio: float,
        device: torch.device,
        k: Optional[int] = None,
        extra_features: list[Callable] = [],
    ):
        # Store information
        self.loss_ratio = loss_ratio
        self.device = device

        # Make the extra_features functions
        extra_features_functions = []
        node_features = 0
        global_features = 0

        for name in extra_features:
            if name == "laplacian_embedding":
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

        # Instanciate the models
        self.diffuser = GnnNodeEdges(
            d=d,
            de=de,
            dy=dy,
            n_heads=n_heads,
            node_features=node_features,
            global_features=global_features,
            n_layers=n_layers,
            natoms=len(VALID_ELEMENTS),
            nbonds=len(TYPE_EDGES),
        )
        self.diffuser = self.diffuser.to(device)

        self.noiser = NoisingModel(
            nodes_distribution,
            edges_distribution,
            diffusion_steps,
        )

        # Make the loss
        self.loss = nn.CrossEntropyLoss()

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

        return total_loss

    def predict(self, batch):
        # unpack the elements of the batch
        N, E, mask = batch  # (bs, n, n_atoms), (bs, n, n, n_edges), (bs,)

        # compute the extra features
        pos_emb, y = self.compute_extra_features(
            E, mask
        )  # (bs, node_features), (bs, n)

        # Compute the prediction of the diffuser
        with torch.no_grad():
            pN, pE, _ = self.diffuser(
                N, E, pos_emb, y, mask
            )  # (bs, n, n_atoms), (bs, n, n, n_edges)

        return pN, pE
