import torch
import torch.nn as nn

from annotix_ml.graphtransf.math.noising import (
    cosine_beta_schedule_discrete,
    sample_discrete_features,
)


class NoisingModel(nn.Module):
    """
    torch module implementing the noise addition of the model.

    Attributes
    ----------
    T: int, number of diffusion steps possible
    natoms: int, the number of valid atoms used.
    nbonds: int, the number of possible bonds.
    n_m: list[float], the nodes distribution of the dataset used (natoms).
    e_m: list[float], the edges distribution of the dataset used (nbonds).
    alphas: torch.Tensor, the constants used to compute the step-wise diffusion matrices
    aphas_bar: torch.Tensor, the constants used to compute the diffusion matrices from 0 to step.

    Methods
    -------
    get_Q_t(t: int) -> tuple[torch.Tensor, torch.Tensor]
    returns the nodes and edges matrices of diffusion from step t-1 to t

    get_Q_bar_t(t: i
    )

    # Load model
    print(f"Loading model from {args.model_checkpoint}...")
    model = DigressMetaArch.load_pretrnt) -> tuple[torch.Tensor, torch.Tensor]
    returns the nodes and edges matrices of diffusion from step 0 to t
    """

    def __init__(
        self,
        nodes_distribution: torch.Tensor,
        edges_distribution: torch.Tensor,
        diffusion_steps: int,
        noise_schedule_type: str = "cosine",
    ):
        super().__init__()
        self.T = diffusion_steps
        self.n_m = nodes_distribution
        self.e_m = edges_distribution
        self.natoms = len(nodes_distribution)
        self.nbonds = len(edges_distribution)

        match noise_schedule_type:
            case "cosine":
                alphas, alphas_bar = cosine_beta_schedule_discrete(diffusion_steps)

            case _:
                raise ValueError(f"{noise_schedule_type} not comprehensible.")

        self.alphas = alphas
        self.alphas_bar = alphas_bar

    def get_Q_t(self, t: int) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = self.alphas[int(t)]
        device = self.alphas.device
        Q_nodes = alpha * torch.eye(len(self.n_m), device=device) + (
            1 - alpha
        ) * self.n_m.unsqueeze(-1).expand(-1, self.natoms)  # (natoms, natoms)

        Q_edges = alpha * torch.eye(len(self.e_m), device=device) + (
            1 - alpha
        ) * self.e_m.unsqueeze(-1).expand(-1, self.nbonds)  # (nbonds, nbonds)

        return Q_nodes.T, Q_edges.T

    def get_Q_bar_t(self, t: int) -> tuple[torch.Tensor, torch.Tensor]:
        alpha_bar = self.alphas_bar[int(t)]
        device = self.alphas_bar.device
        Q_bar_nodes = alpha_bar * torch.eye(len(self.n_m), device=device) + (
            1 - alpha_bar
        ) * self.n_m.unsqueeze(-1).expand(-1, self.natoms)  # (natoms, natoms)

        Q_bar_edges = alpha_bar * torch.eye(len(self.e_m), device=device) + (
            1 - alpha_bar
        ) * self.e_m.unsqueeze(-1).expand(-1, self.nbonds)  # (nbonds, nbonds)

        return Q_bar_nodes.T, Q_bar_edges.T

    def get_posterior(
        self, N: torch.Tensor, E: torch.Tensor, t: int, eps: float = 1e-6
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the posterior distribution of nodes and edges at step t-1 given step t.
        According to the formula of Vignac it is :
        Z^t @ (Q^t)' * Z^0 @ Q_bar^(t-1) / Z^0 @ Q_bar^t @ (Z^t)'

        Args:
        - N: torch.Tensor, the nodes one-hot encoded vector at step t.
        - E: torch.Tensor, the edges one-hot encoded vector at step t.
        - t: int, the current time step.
        - eps: float = 1e-6, the value to add to the sum and avoid zero div.

        Returns:
        The posterior probabilities for nodes and edges at step t-1. The proba
        are computed for each possible values of N^0 and E^0 making them resp
        (bs, n, natoms, natoms) and (bs, n, n, nbonds, nbonds) tensors.
        """
        Q_n, Q_e = self.get_Q_t(t)  #  (natoms, natoms), (nbonds, nbonds)
        Q_bar_n, Q_bar_e = self.get_Q_bar_t(t)  #  (natoms, natoms), (nbonds, nbonds)
        Q_bar_nb, Q_bar_eb = self.get_Q_bar_t(
            t - 1
        )  #  (natoms, natoms), (nbonds, nbonds)

        # Compute the posterior distribution of Nodes
        # Start with the numerator of the equation
        # N @ Q_n.T: (bs, n, natoms)
        pN_numerator = (N @ Q_n.T).unsqueeze(2)  # (bs, n, 1, natoms)
        pN_numerator = pN_numerator * Q_bar_nb.unsqueeze(0).unsqueeze(
            0
        )  # (bs, n, natoms, natoms)

        # Continue with the denominator
        pN_denominator = (N @ Q_bar_n.T).unsqueeze(-1)  # (bs, n, natoms, 1)

        # Compute the posterior
        pN_posterior = pN_numerator / (pN_denominator + eps)

        # Compute the posterior distribution of Edges
        pE_numerator = (E @ Q_e.T).unsqueeze(3)  # (bs, n, n, 1, nbonds)
        pE_numerator = pE_numerator * Q_bar_eb.unsqueeze(0).unsqueeze(0).unsqueeze(
            0
        )  # (bs, n, n, nbonds, nbonds)

        # Compute the denominator
        pE_denominator = (E @ Q_bar_e.T).unsqueeze(-1)  # (bs, n, n, nbonds, 1)

        # Compute the posterior
        pE_posterior = pE_numerator / (pE_denominator + eps)

        return (
            pN_posterior,
            pE_posterior,
        )  # (bs, n, natoms, natoms) and (bs, n, n, nbonds, nbonds)

    @torch.no_grad
    def forward(
        self, N: torch.Tensor, E: torch.Tensor, node_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Noise the nodes and edges matrices of a given batch. A random t is
        sampled for each graph and the edges and nodes are noised with the
        corresponding matrices.

        Args:
        - N: torch.Tensor, the nodes one-hot encoded vector.
        - E: torch.Tensor, the edges one-hot encoded vector.
        - node_mask: torch.Tensor, the mask vector.

        Returns:
        Noised nodes and edges, as well as the t used for noising.
        """
        # N (bs, n, natoms)
        # E (bs, n, n, nbonds)

        bs = N.shape[0]
        device = N.device
        sampled_t = torch.randint(1, self.T, (bs,), device=device)
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
        Compute the noised graph at a given step t.

        Args:
        - N: torch.Tensor, the nodes one-hot encoded vector.
        - E: torch.Tensor, the edges one-hot encoded vector.
        - node_mask: torch.Tensor | None, the mask vector.
        - t: int | torch.Tensor, the time step(s) to noise to.

        Returns: Noised nodes and edges.
        """
        # Handle unbatched input
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

        # Now N is (bs, n, natoms), E is (bs, n, n, nbonds)
        bs = N.shape[0]
        device = N.device
        type_tensor = N.dtype

        if isinstance(t, int):
            t_tensor = torch.full((bs,), t, dtype=torch.long, device=device)
        else:
            t_tensor = t.to(device)
            if t_tensor.dim() == 0:
                t_tensor = t_tensor.unsqueeze(0)

        # Make the matrices & stack them
        Q_matrices = [self.get_Q_bar_t(int(t_val.item())) for t_val in t_tensor]
        Q_bar_nodes, Q_bar_edges = zip(
            *Q_matrices
        )  # (natoms, natoms), (nbonds, nbonds)

        Q_nodes = torch.stack(Q_bar_nodes).to(device)  # (bs, natoms, natoms)
        Q_edges = (
            torch.stack(Q_bar_edges).unsqueeze(1).to(device)
        )  # (bs, 1, nbonds, nbonds)

        # Do the noising
        N = N @ Q_nodes  # (bs, n, natoms)
        E = E @ Q_edges  # (bs, n, n, nbonds)

        # Sample discrete features
        N, E = sample_discrete_features(N, E, node_mask)

        # Cast back to the original type
        N = N.to(type_tensor)
        E = E.to(type_tensor)

        if is_unbatched:
            return N.squeeze(0), E.squeeze(0)

        return N, E

    def move_to(self, device: torch.device) -> None:
        """
        Move the noiser to the specified device.

        Args:
        - device: torch.device, the device to move the noiser to.
        """
        self.alphas = self.alphas.to(device)
        self.alphas_bar = self.alphas_bar.to(device)
        self.n_m = self.n_m.to(device)
        self.e_m = self.e_m.to(device)
