import torch

def cosine_beta_schedule_discrete(timesteps, s=0.008):
    """Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ."""
    steps = timesteps + 2
    x = torch.arange(steps) # (timesteps + 2)

    alphas_cumprod = torch.cos(0.5 * torch.pi * ((x / steps) + s) / (1 + s)) ** 2 # (timesteps + 2)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0] # (timesteps + 2)
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1] # (timesteps + 1)
    betas = 1 - alphas # (timesteps + 1)
    return betas
