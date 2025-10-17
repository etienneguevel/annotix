import torch
import torch.nn.functional as F

def cosine_beta_schedule_discrete(timesteps: int, s=0.008):
    """
    Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ.
    alpha_bar_t = f(t)/f(0), f(t) = cos(0.5*pi((t/steps)+s)/(1+s))
    beta_t = 1 - alpha_bar_t / alpha_bar_t-1
    """
    steps = timesteps + 2
    x = torch.arange(steps) # (timesteps + 2)

    # Compute raw alphas according to f
    alphas_cumprod = torch.cos(0.5 * torch.pi * ((x / steps) + s) / (1 + s)) ** 2 # (timesteps + 2)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0] # (timesteps + 2)
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1] # (timesteps + 1)

    # Compute betas and limit their max values
    betas = 1 - alphas
    betas = 1 - torch.clamp(betas, min=0, max=0.999) # limit the max value of betas to prevent irregularities

    # Recompute the alphas with the new betas values
    alphas = 1 - betas
    log_alpha = torch.log(alphas)
    log_alpha_bar = torch.cumsum(log_alpha, dim=0)
    alphas_bar = torch.exp(log_alpha_bar)

    return alphas, alphas_bar

def sample_discrete_features(
    probX: torch.Tensor, 
    probE: torch.Tensor, 
    node_mask: torch.Tensor
):
    """
    Sample features from multinomial distribution with given probabilities.
    Args:
        - probX: bs, n, natoms        node features
        - probE: bs, n, n, nbonds     edge features
        - node_mask: bs, n            binary mask for nodes
    """
    bs, n, natoms = probX.shape
    nbonds = probE.shape[-1]
    # Noise X
    # The masked rows should define probability distributions as well
    node_mask = (node_mask == 1) # convert mask to bool type
    probX[~node_mask] = 1 / probX.shape[-1]

    # Flatten the probability tensor to sample with multinomial
    probX = probX.reshape(bs * n, -1)  # (bs * n, natoms)

    # Sample X
    X_t = probX.multinomial(1)  # (bs * n, 1)
    X_t = X_t.reshape(bs, n)  # (bs, n)

    # Noise E
    # The masked rows should define probability distributions as well
    inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
    diag_mask = torch.eye(n).unsqueeze(0).expand(bs, -1, -1) # (bs, n, n)

    probE[inverse_edge_mask] = 1 / probE.shape[-1]
    probE[diag_mask.bool()] = 1 / probE.shape[-1]

    probE = probE.reshape(bs * n * n, -1)  # (bs * n * n, de_out)

    # Sample E
    E_t = probE.multinomial(1).reshape(bs, n, n)  # (bs, n, n)
    E_t = torch.triu(E_t, diagonal=1) # (bs, n, n)
    E_t = E_t + torch.transpose(E_t, 1, 2) # (bs, n, n)

    # One-hot encode the noised X and E
    X_t = F.one_hot(X_t, natoms)
    E_t = F.one_hot(E_t, nbonds)

    return X_t, E_t
