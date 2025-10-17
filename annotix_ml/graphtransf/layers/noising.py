import torch
import torch.nn as nn

from annotix_ml.graphtransf.math.noising import (
    cosine_beta_schedule_discrete, sample_discrete_features
)

class NoisingModel(nn.Module):
    def __init__(
        self,
        nodes_distribution: list[float],
        edges_distribution: list[float],
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

    def get_Q_t(self, t):
        alpha = self.alphas[t]
        Q_nodes = (
            alpha * torch.eye(len(self.n_m)) + (1 - alpha) * self.n_m.unsqueeze(-1).expand(-1, self.natoms)
        ) # (natoms, natoms)

        Q_edges = (
            alpha * torch.eye(len(self.e_m)) + (1 - alpha) * self.e_m.unsqueeze(-1).expand(-1, self.nbonds)
        ) # (nbonds, nbonds)

        return Q_nodes.T, Q_edges.T
    
    def get_Q_bar_t(self, t):
        alpha_bar = self.alphas_bar[t]
        Q_bar_nodes = (
            alpha_bar * torch.eye(len(self.n_m)) + (1 - alpha_bar) * self.n_m.unsqueeze(-1).expand(-1, self.natoms)
        ) # (natoms, natoms)

        Q_bar_edges = (
            alpha_bar * torch.eye(len(self.e_m)) + (1 - alpha_bar) * self.e_m.unsqueeze(-1).expand(-1, self.nbonds)
        ) # (nbonds, nbonds)

        return Q_bar_nodes.T, Q_bar_edges.T
    
    @torch.no_grad
    def forward(self, N, E, node_mask):
        """
        """
        # N (bs, n, natoms)
        # E (bs, n, n, nbonds)
        bs = N.shape[0]
        device = N.device
        sampled_t = torch.randint(1, self.T, (bs,))

        # Make the matrices & stack them
        Q_matrices = [self.get_Q_bar_t(t) for t in sampled_t]
        Q_bar_nodes, Q_bar_edges = zip(*Q_matrices) # (natoms, natoms), (nbonds, nbonds)

        Q_nodes = torch.stack(Q_bar_nodes).to(device) # (bs, natoms, natoms)
        Q_edges = torch.stack(Q_bar_edges).unsqueeze(1).to(device) # (bs, 1, nbonds, nbonds)

        # Do the noising
        N = N @ Q_nodes # (bs, n, natoms)
        E = E @ Q_edges # (bs, n, n, nbonds)

        # TODO : should I noise somewhere ?
        # Doesn't seem necessary for N, what about E ?
        N, E = sample_discrete_features(N, E, node_mask)  

        return N, E, node_mask