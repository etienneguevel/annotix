import json
import os
import shutil
from argparse import ArgumentParser
from collections import defaultdict

import torch
import wandb
from rdkit.RDLogger import DisableLog  # pyright: ignore[reportAttributeAccessIssue]
from omegaconf import OmegaConf
from torch.linalg import LinAlgError
from annotix_ml.graphtransf.train.memory_tracker import LayerMemoryTracker
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

import annotix_ml.distributed as dist
from annotix_ml.distributed.pipeline_parallelism import save_checkpoint
from annotix_ml.graphtransf.arch.digress_meta_arch import DigressMetaArch
from annotix_ml.graphtransf.data.datacollator import collateGraph, collateGraphStatic
from functools import partial
from annotix_ml.graphtransf.data.data_utils import (
    batch_graph_to_smiles,
    batch_graph_to_smiles_digress,
)
from annotix_ml.graphtransf.data.atoms_data import TYPE_EDGES
from annotix_ml.graphtransf.data.loaders import make_datasets
from annotix_ml.graphtransf.data.samplers import InfiniteSampler
from annotix_ml.graphtransf.math.metrics import (
    compute_metrics,
    compute_training_metrics,
)
from annotix_ml.graphtransf.train.setup import setup
from annotix_ml.distributed import enable


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--project-name", type=str, required=False)
    parser.add_argument("--save-path", type=str, required=False)
    parser.add_argument("--extra-features", type=str, required=False)
    parser.add_argument("--batch-size", type=int, required=False)
    parser.add_argument("--num-train-steps", type=int, required=False)

    return parser.parse_args()


def do_eval(
    model: DigressMetaArch,
    eval_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    DisableLog("rdApp.*")
    metrics = defaultdict(list)
    for batch in tqdm(
        eval_loader, desc="evaluation", disable=not dist.is_main_process()
    ):
        # Move the elements to the device of interest
        batch = [v.to(device) if isinstance(v, torch.Tensor) else v for v in batch]

        # Unpack the elements
        nodes, edges, mask = batch

        # Noise the elements
        nodes_noised, edges_noised, sampled_t = model.noiser(nodes, edges, mask)

        # Compute the predictions
        pN, pE = model.forward(
            nodes_noised, edges_noised, mask, sampled_t
        )  # (bs, n, n_atoms), (bs, n, n, n_edges)

        if pN is None or pE is None:
            return None

        # Make the prediction graph
        N_ = torch.nn.functional.one_hot(
            pN.argmax(-1), num_classes=pN.shape[-1]
        )  # (bs, n, n_atoms)
        E_ = torch.nn.functional.one_hot(
            pE.argmax(-1), num_classes=pE.shape[-1]
        )  # (bs, n, n, n_edges)

        # Convert the graph to smiles
        true_smiles = batch_graph_to_smiles(
            nodes, edges, mask, model.valid_elements
        )  # (bs,)
        pred_smiles = batch_graph_to_smiles(N_, E_, mask, model.valid_elements)  # (bs,)

        # Compute the metrics
        batch_metrics = compute_metrics(
            pred_smiles,
            true_smiles,
            pN,
            pE,
            nodes,
            edges,
            mask,
        )

        # Compute accuracy per node type
        target_nodes = nodes.argmax(-1)
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
        target_edges = edges.argmax(-1)
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

    return final_metrics


def generate_samples(
    model: DigressMetaArch, num_samples: int, num_nodes_dist: torch.Tensor
) -> tuple[float, float, list[str]]:
    # Sample from the distribution
    n = (
        num_nodes_dist.unsqueeze(0).expand((num_samples, -1)).multinomial(1).squeeze(-1)
        + 1
    )

    # Generate the graphs
    print("Generating graphs for validity computation...")
    try:
        gen_N, gen_E, gen_mask = model.generate(
            num_samples=n, max_nodes=n.max(), progress_bar=True
        )

        # Convert to smiles
        gen_smiles = batch_graph_to_smiles(gen_N, gen_E, gen_mask, model.valid_elements)
        valid_smiles = [s for s in gen_smiles if s]

        # Convert to smiles with Digress method
        gen_smiles_digress = batch_graph_to_smiles_digress(
            gen_N, gen_E, gen_mask, model.valid_elements
        )
        valid_smiles_digress = [s for s in gen_smiles_digress if s]

        # Compute the validity
        validity = len(valid_smiles) / len(gen_smiles)
        validity_digress = len(valid_smiles_digress) / len(gen_smiles)

    except LinAlgError:
        validity = 0
        validity_digress = 0
        valid_smiles = []

    return validity, validity_digress, valid_smiles


def train(cfg):
    # Initialize wandb
    if dist.is_main_process():
        # Constrain wandb to only see the current GPU for system metrics
        print(dist._MAIN_RANK)

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
        cfg.dataset.val_tag,
        verbose=dist.is_main_process(),
    )

    # Determine the collation function and distributed data rank/size
    data_rank = dist.get_global_rank()
    data_size = dist.get_global_size()

    if dist.is_enabled():
        if cfg.train.get("distributed") == "pipeline":
            # In Pipeline Parallelism, all ranks in the same pipeline
            # (which is the whole world here) must see the same data.
            data_rank = 0
            data_size = 1
            print(
                f"Pipeline Parallelism detected: setting data_rank={data_rank}, data_size={data_size} to synchronize input/targets"
            )

        # Derive n_max from the distribution of number of atoms
        n_max = len(train_dataset.num_atoms_dist)
        collate_fn = partial(collateGraphStatic, n_max=n_max)
        print(f"Using static shape data collator with n_max={n_max}")
    else:
        collate_fn = collateGraph

    # Make the DataLoaders
    sample_count = len(train_dataset)
    shuffle = True
    seed = cfg.train.seed
    advance = 0

    sampler_type = InfiniteSampler(
        sample_count=sample_count,
        shuffle=shuffle,
        seed=seed,
        start=data_rank,
        step=data_size,
        advance=advance,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        sampler=sampler_type,
        collate_fn=collate_fn,
        num_workers=cfg.train.num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.valid.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
    )

    # Define the models
    digress = DigressMetaArch.init_from_cfg(
        cfg,
        device,
        train_dataset.valid_elements,
        train_dataset.nodes_distribution,
        train_dataset.edges_distribution,
        max_weight=train_dataset.max_weight,
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
    num_param = sum([p.numel() for p in digress.diffuser.parameters()])
    print("=" * 50)
    print("DIFFUSER MODEL ARCHITECTURE")
    print("=" * 50)
    print(f"Num parameters: {num_param / 1e6:.2f}")
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
    print(f"Max weight in dataset: {train_dataset.max_weight}")
    print("=" * 50 + "\n")

    # Define the metrics
    metrics = defaultdict(list)
    train_metrics = defaultdict(list)

    # Prepare the save path
    save_path = cfg.train.save_path
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)

    os.makedirs(save_path)

    # Setup for distributed training
    if cfg.train.get("distributed") is not None:
        example_batch = next(iter(train_loader))
        digress._setup_distributed(
            cfg.train.distributed, cfg.train.num_microbatches, example_batch
        )

    # Make the optimizer
    optimizer = torch.optim.AdamW(
        digress.diffuser.parameters(),
        lr=cfg.train.starting_learning_rate,
        amsgrad=True,
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
    pbar = tqdm(
        enumerate(train_loader), desc="Training", disable=not dist.is_main_process()
    )

    # Setup per-layer memory tracking on each process
    mem_tracker = LayerMemoryTracker(digress.diffuser, device)

    start_train_time = time.time()
    for i, batch in pbar:
        batch_start_time = time.time()
        # Make the model in train mode
        digress.diffuser.train()

        # zero grad before forward pass
        optimizer.zero_grad()

        # Reset memory tracker for this step
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        mem_tracker.reset()

        # Move the batch to the correct device
        batch = [v.to(device) if isinstance(v, torch.Tensor) else v for v in batch]
        N, E, mask = batch

        # Do the forward and loss computation
        try:
            pN, pE, loss = digress.forward_backward(N, E, mask)

        except LinAlgError:
            print("LinAlgError in forward or digress_loss")
            continue

        # If loss is None (on non-last ranks in PP), we skip logging
        if loss is not None:
            epoch_metrics = compute_training_metrics(
                pN, pE, N, E, mask, digress.valid_elements
            )

            for k, v in epoch_metrics.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                train_metrics[k].append(v)

            # Update the progress bar
            lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=loss.item(), lr=lr, **epoch_metrics)

            # Log training metrics to wandb
            train_log = {
                "train/loss": loss.item(),
                "train/learning_rate": lr,
                "train/epoch_time": time.time() - batch_start_time,
                "train/total_time": time.time() - start_train_time,
                "step": i,
            }

            # Log per-layer memory metrics from hooks
            train_log.update(mem_tracker.get_metrics())

            for k, v in epoch_metrics.items():
                train_log[f"train/{k}"] = v

            wandb.log(train_log)

        # Update the parameters
        optimizer.step()

        # Step the scheduler
        if scheduler is not None:
            scheduler.step()

        # Start the evaluation
        if (i % cfg.valid.num_eval_steps == 0) & (i > 0):
            # Make the model in eval mode
            digress.diffuser.eval()

            # Do the evaluation
            with torch.no_grad():
                eval_metrics = do_eval(digress, valid_loader, device)

                if eval_metrics:
                    validity, validity_digress, valid_smiles = generate_samples(
                        digress,
                        cfg.valid.num_samples,
                        train_dataset.num_atoms_dist,
                    )
                    eval_metrics["gen_validity"] = validity
                    eval_metrics["gen_validity_digress"] = validity_digress

                    for k, v in eval_metrics.items():
                        metrics[k].append(v)

                    if dist.is_main_process():
                        print(f"Evaluation Metrics: {eval_metrics}")

                        # Log evaluation metrics to wandb
                        eval_log = {"step": i}
                        for k, v in eval_metrics.items():
                            eval_log[f"eval/{k}"] = v

                        wandb.log(eval_log)

                        # Save the valid smiles
                        with open(
                            os.path.join(cfg.train.save_path, f"valid_smiles_{i}.txt"),
                            "w",
                        ) as f:
                            for s in valid_smiles:
                                f.write(f"{s}\n")

        # Save the model
        if (i % cfg.train.save_steps == 0) & (i > 0):
            if cfg.train.get("distributed") == "pipeline":
                save_checkpoint(
                    digress.diffuser,
                    optimizer,
                    os.path.join(cfg.train.save_path, f"checkpoint_{i}"),
                )

            else:
                torch.save(
                    digress.diffuser.state_dict(),
                    os.path.join(cfg.train.save_path, f"{i}.pt"),
                )

        # Stop the training when the desired number of steps has been reached
        if i >= cfg.train.num_train_steps:
            break

    mem_tracker.remove_hooks()

    # Save the final model
    if cfg.train.get("distributed") == "pipeline":
        save_checkpoint(
            digress.diffuser,
            optimizer,
            os.path.join(cfg.train.save_path, "final"),
        )

    else:
        torch.save(
            digress.diffuser.state_dict(),
            os.path.join(cfg.train.save_path, "final.pt"),
        )

    # Save the metrics
    with open(os.path.join(cfg.train.save_path, "val_metrics.json"), "w") as f:
        json.dump(metrics, f)

    with open(os.path.join(cfg.train.save_path, "train_metrics.json"), "w") as f:
        json.dump(train_metrics, f)


def main():
    args = get_args()
    cfg = setup(args)

    if cfg.train.distributed == "pipeline":
        main_rank = "last"

    else:
        main_rank = "first"

    enable(overwrite=True, main_rank=main_rank)
    train(cfg)


if __name__ == "__main__":
    main()
