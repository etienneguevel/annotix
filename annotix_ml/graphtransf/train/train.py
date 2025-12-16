import json
import os
import shutil
from argparse import ArgumentParser
from collections import defaultdict

import torch
import wandb
from rdkit.RDLogger import DisableLog  # pyright: ignore[reportAttributeAccessIssue]
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from annotix_ml.graphtransf.arch.digress_meta_arch import DigressMetaArch
from annotix_ml.graphtransf.data.datacollator import collateGraph
from annotix_ml.graphtransf.data.data_utils import batch_graph_to_smiles
from annotix_ml.graphtransf.data.atoms_data import TYPE_EDGES
from annotix_ml.graphtransf.data.loaders import make_datasets
from annotix_ml.graphtransf.data.samplers import InfiniteSampler
from annotix_ml.graphtransf.math.metrics import compute_metrics
from annotix_ml import ROOT


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def do_eval(model: DigressMetaArch, eval_loader: DataLoader, device: torch.device):
    DisableLog("rdApp.*")
    metrics = defaultdict(list)
    for batch in tqdm(eval_loader, desc="evaluation"):
        # Move the elements to the device of interest
        batch = tuple(el.to(device) for el in batch)

        # Unpack the elements
        N, E, mask = batch  # (bs, n, n_atoms), (bs, n, n, n_edges), (bs,)

        # Compute the predictions
        pN, pE = model.forward(batch)  # (bs, n, n_atoms), (bs, n, n, n_edges)

        # Make the prediction graph
        N_ = torch.nn.functional.one_hot(
            pN.argmax(-1), num_classes=pN.shape[-1]
        )  # (bs, n, n_atoms)
        E_ = torch.nn.functional.one_hot(
            pE.argmax(-1), num_classes=pE.shape[-1]
        )  # (bs, n, n, n_edges)

        # Convert the graph to smiles
        true_smiles = batch_graph_to_smiles(N, E, mask, model.valid_elements)  # (bs,)
        pred_smiles = batch_graph_to_smiles(N_, E_, mask, model.valid_elements)  # (bs,)

        # Compute the metrics
        batch_metrics = compute_metrics(
            pred_smiles,
            true_smiles,
            pN,
            pE,
            N,
            E,
            mask,
        )

        # Compute accuracy per node type
        target_nodes = N.argmax(-1)
        pred_nodes = pN.argmax(-1)
        valid_mask = mask.bool()

        target_nodes_flat = target_nodes[valid_mask]
        pred_nodes_flat = pred_nodes[valid_mask]

        for i, atom_type in enumerate(model.valid_elements):
            atom_mask = target_nodes_flat == i
            if atom_mask.sum() > 0:
                acc = (pred_nodes_flat[atom_mask] == i).float().mean().item()
                batch_metrics[f"accuracy_node_{atom_type}"] = [acc]

        # Compute accuracy per edge type
        target_edges = E.argmax(-1)
        pred_edges = pE.argmax(-1)
        edge_mask = mask.unsqueeze(2) * mask.unsqueeze(1)
        edge_mask = edge_mask.bool()

        target_edges_flat = target_edges[edge_mask]
        pred_edges_flat = pred_edges[edge_mask]

        if len(target_edges_flat) > 0:
            acc_edge = (pred_edges_flat == target_edges_flat).float().mean().item()
            batch_metrics["accuracy_edge_global"] = [acc_edge]

        for i, edge_type in enumerate(TYPE_EDGES):
            edge_type_mask = target_edges_flat == i
            if edge_type_mask.sum() > 0:
                acc = (pred_edges_flat[edge_type_mask] == i).float().mean().item()
                batch_metrics[f"accuracy_edge_{str(edge_type)}"] = [acc]

        # Accumulate metrics
        for k, v in batch_metrics.items():
            metrics[k].extend(v)

    # Average and print metrics
    final_metrics = {}
    for k, v in metrics.items():
        if len(v) > 0:
            final_metrics[k] = sum(v) / len(v)
        else:
            final_metrics[k] = 0.0

    # Compute the validity of the generated graphs
    # We use the max number of nodes in the validation set
    max_n = 0
    for batch in eval_loader:
        N, _, _ = batch
        max_n = max(max_n, N.shape[1])

    # Generate the graphs
    print("Generating graphs for validity computation...")
    gen_N, gen_E, gen_mask = model.generate(
        batch_size=eval_loader.batch_size, max_nodes=max_n, progress_bar=True
    )

    # Convert to smiles
    gen_smiles = batch_graph_to_smiles(gen_N, gen_E, gen_mask, model.valid_elements)

    # Compute the validity
    validity = sum([1 for s in gen_smiles if s is not None]) / len(gen_smiles)
    final_metrics["gen_validity"] = validity
    print(f"Evaluation Metrics: {final_metrics}")

    return final_metrics


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

    train_dataset, valid_dataset = make_datasets(
        cfg.dataset.data_path,
        cfg.dataset.smile_column,
        cfg.dataset.split_column,
    )

    # Make the DataLoaders
    sample_count = len(train_dataset)
    shuffle = True
    seed = cfg.train.seed
    advance = 0

    sampler_type = InfiniteSampler(
        sample_count=sample_count,
        shuffle=shuffle,
        seed=seed,
        advance=advance,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        sampler=sampler_type,
        collate_fn=collateGraph,
        num_workers=cfg.train.num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.valid.batch_size,
        collate_fn=collateGraph,
        shuffle=False,
    )

    # Define the models
    digress = DigressMetaArch.init_from_cfg(
        cfg,
        device,
        train_dataset.valid_elements,
        train_dataset.nodes_distribution,
        train_dataset.edges_distribution,
    )
    digress.valid_elements = train_dataset.valid_elements

    # Log model dimensions
    print("\n" + "=" * 50)
    print("MODEL DIMENSIONS")
    print("=" * 50)
    print(f"Hidden dimension (d): {cfg.model.d}")
    print(f"Edge dimension (de): {cfg.model.de}")
    print(f"Global features dimension (dy): {cfg.model.dy}")
    print(f"Number of attention heads: {cfg.model.n_heads}")
    print(f"Number of layers: {cfg.model.n_layers}")
    print(f"Number of atom types (natoms): {digress.natoms}")
    print(f"Number of bond types (nbonds): {digress.nbonds}")
    if hasattr(cfg.model, "extra_features") and cfg.model.extra_features:
        print(f"Extra features: {cfg.model.extra_features}")
    print("=" * 50 + "\n")

    # Print the diffuser model architecture
    print("=" * 50)
    print("DIFFUSER MODEL ARCHITECTURE")
    print("=" * 50)
    print(digress.diffuser)
    print("=" * 50 + "\n")

    # Log dataset information
    print("=" * 50)
    print("DATASET INFORMATION")
    print("=" * 50)
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")
    print(f"Valid elements: {train_dataset.valid_elements}")
    print(f"Node distribution: {train_dataset.nodes_distribution.tolist()}")
    print(f"Edge distribution: {train_dataset.edges_distribution.tolist()}")
    print("=" * 50 + "\n")

    # Define the metrics
    metrics = defaultdict(list)
    train_metrics = defaultdict(list)

    # Prepare the save path
    save_path = cfg.train.save_path
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)

    os.makedirs(save_path)

    # Make the losses and the optimizer
    optimizer = torch.optim.AdamW(
        digress.diffuser.parameters(), lr=cfg.train.starting_learning_rate
    )

    # Define the scheduler
    if cfg.train.learning_rate_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.train.num_train_steps,
            eta_min=cfg.train.final_learning_rate,
        )
    else:
        scheduler = None

    # Start the training loop
    pbar = tqdm(enumerate(train_loader), desc="Training")
    for i, batch in pbar:
        # Make the model in train mode
        digress.diffuser.train()

        # Move the batch to the correct device
        batch = tuple(el.to(device) for el in batch)

        # Do the forward backward loop
        loss, acc = digress.forward_backward(batch)
        for k, v in acc.items():
            train_metrics[k].append(v)

        # Update the parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Update the progress bar
        lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(loss=loss.item(), lr=lr, **acc)  # pyright: ignore[reportArgumentType]

        # Log training metrics to wandb
        train_log = {"train/loss": loss.item(), "train/learning_rate": lr, "step": i}
        for k, v in acc.items():
            train_log[f"train/{k}"] = v

        wandb.log(train_log)

        # Remove the batch from memory
        del batch

        if i % cfg.valid.num_eval_steps == 0:
            # Make the model in eval mode
            digress.diffuser.eval()

            # Do the evaluation
            with torch.no_grad():
                eval_metrics = do_eval(digress, valid_loader, device)
                for k, v in eval_metrics.items():
                    metrics[k].append(v)

                # Log evaluation metrics to wandb
                eval_log = {"step": i}
                for k, v in eval_metrics.items():
                    eval_log[f"eval/{k}"] = v

                wandb.log(eval_log)

        if i % cfg.train.save_steps == 0:
            torch.save(
                digress.diffuser.state_dict(),
                os.path.join(cfg.train.save_path, f"{i}.pt"),
            )

        if i >= cfg.train.num_train_steps:
            break

    # Save the final model
    torch.save(
        digress.diffuser.state_dict(), os.path.join(cfg.train.save_path, "final.pt")
    )

    # Save the metrics
    with open(os.path.join(cfg.train.save_path, "val_metrics.json"), "w") as f:
        json.dump(metrics, f)

    with open(os.path.join(cfg.train.save_path, "train_metrics.json"), "w") as f:
        json.dump(train_metrics, f)


def main():
    args = get_args()

    # Load default config
    default_cfg_path = ROOT / "graphtransf" / "configs" / "default_config.yaml"
    default_cfg = OmegaConf.load(default_cfg_path)

    # Load user config
    user_cfg = OmegaConf.load(args.config)

    # Merge configs
    cfg = OmegaConf.merge(default_cfg, user_cfg)

    train(cfg)


if __name__ == "__main__":
    main()
