import torch
import torch.nn as nn

from annotix_ml.graphtransf.math.noising import cosine_beta_schedule_discrete

class NoisingModel(nn.module):
    def __init__(
        self,
        nodes_distribution: list[float],
        edges_distribution: list[float],
        diffusion_steps: int,
        noise_schedule_type: str = "cosine",
    ):
        super().__init__()
        self.x_m = nodes_distribution
        self.e_m = edges_distribution
        self.T = diffusion_steps

        match noise_schedule_type:
            case "cosine":
                self._make_noise_matrices(diffusion_steps)
            
            case _:
                raise ValueError(f"{noise_schedule_type} not comprehensible.")
        
    
    def _make_noise_matrices(self, diffusion_steps):


    
    @torch.no_grad
    def forward(self, N, E):
        ...


    