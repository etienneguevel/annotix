import torch
import wandb
from omegaconf import OmegaConf


def train(cfg):
    # Initialize wandb
    wandb.init(
        project=cfg.run.project,
        name=cfg.run.name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # Select the device for the training
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}\n")

    # make datasets
