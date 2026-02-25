import json
import os
import time
from argparse import ArgumentParser
from collections import defaultdict

import torch
import wandb
from omegaconf import OmegaConf
from torch.linalg import LinAlgError
from torch.utils.data import DataLoader
from tqdm import tqdm

import annotix_ml.distributed as dist
from annotix_ml.distributed.pipeline_parallelism import save_checkpoint
from annotix_ml.data.samplers import InfiniteSampler
from annotix_ml.graphtransf.math.metrics import (
    compute_training_metrics,
)
from annotix_ml.train.eval import do_eval, generate_samples
from annotix_ml.train.setup import setup, setup_model_mode


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--project-name", type=str, required=False)
    parser.add_argument("--save-path", type=str, required=False)
    parser.add_argument("--extra-features", type=str, required=False)
    parser.add_argument("--batch-size", type=int, required=False)
    parser.add_argument("--num-train-steps", type=int, required=False)
    parser.add_argument("--num-diffusion-steps", type=int, required=False)
    parser.add_argument("--distributed-strat", type=str, required=False)

    return parser.parse_args()


def train(cfg):
    # Select the device for the training
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}\n")

    # Determine the collation function and distributed data rank/size
    data_rank = dist.get_global_rank()
    data_size = dist.get_global_size()

    # Make the training objects from the cfg mode (graphtransf or spec2mol)
    train_dataset, valid_dataset, arch, collate_fn = setup_model_mode(cfg)

    # Prepare the save path & check if there is already some existing checkpoints
    save_path = cfg.train.save_path
    last_epoch = 0
    checkpoint_path = None

    if not os.path.isdir(save_path):
        if dist.is_main_process():
            os.makedirs(save_path)
            # Save a copy of the config file
            OmegaConf.save(config=cfg, f=os.path.join(save_path, "config.yaml"))

    if dist.is_enabled():
        torch.distributed.barrier()

    # Check if there are already model weights in the saving path
    model_checkpoints = [f for f in os.listdir(save_path) if f.endswith(".pt")]
    if len(model_checkpoints) > 0:
        last_checkpoint = sorted(model_checkpoints, key=lambda y: int(y.split(".")[0]))[
            -1
        ]
        checkpoint_path = os.path.join(save_path, last_checkpoint)
        last_epoch = int(last_checkpoint.split(".")[0])

        print(f"Checkpoints found, loading from the weights at {checkpoint_path}.")
        digress = arch.load_pretrained(
            cfg,
            device,
            checkpoint_path,
            train_dataset.valid_elements,
            train_dataset.nodes_distribution,
            train_dataset.edges_distribution,
            train_dataset.max_weight,
        )

    else:
        # Define the model
        print("No checkpoints found, init from zero.")
        digress = arch.init_from_cfg(
            cfg,
            device,
            train_dataset.valid_elements,
            train_dataset.nodes_distribution,
            train_dataset.edges_distribution,
            max_weight=train_dataset.max_weight,
        )

    if dist.is_main_process():
        print(f"Logging with main rank as : {dist._MAIN_RANK}")

        wandb_id_file = os.path.join(save_path, "wandb_run_id.txt")
        if last_epoch > 0 and os.path.isfile(wandb_id_file):
            with open(wandb_id_file) as f:
                wandb_run_id = f.read().strip()
            wandb.init(
                project=cfg.run.project,
                name=cfg.run.name,
                config=OmegaConf.to_container(cfg, resolve=True),
                id=wandb_run_id,
                resume="allow",
            )
        else:
            wandb.init(
                project=cfg.run.project,
                name=cfg.run.name,
                config=OmegaConf.to_container(cfg, resolve=True),
            )
            with open(wandb_id_file, "w") as f:
                f.write(wandb.run.id)

    # Make the DataLoaders
    sample_count = len(train_dataset)
    shuffle = True
    seed = cfg.train.seed
    advance = (
        ((last_epoch + 1) * cfg.train.batch_size) % len(train_dataset)
        if (last_epoch > 0 and not shuffle)
        else 0
    )

    # If the training is pipeline dist -> train and valid bs need to be =
    # TODO : find a way to rm that dependency
    if cfg.train.get("distributed") == "pipeline":
        cfg.valid.batch_size = cfg.train.batch_size

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
    num_atoms_dist = train_dataset.num_atoms_dist.tolist()
    bar_width = 30
    max_prob = max(num_atoms_dist) if num_atoms_dist else 1.0
    print("Number of atoms distribution:")
    for idx, prob in enumerate(num_atoms_dist):
        n_atoms = idx + 1
        bar_len = int(prob / max_prob * bar_width)
        bar = "#" * bar_len
        print(f"  {n_atoms:3d} atoms | {bar:<{bar_width}} {prob:.4f}")
    print(f"Max weight in dataset: {train_dataset.max_weight}")
    print("=" * 50 + "\n")

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

    # Restore optimizer state when resuming from a checkpoint that contains it
    if checkpoint_path is not None:
        saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if isinstance(saved, dict) and "optimizer" in saved:
            optimizer.load_state_dict(saved["optimizer"])
            print(f"Optimizer state restored from {checkpoint_path}.")

    # Define the scheduler
    if cfg.train.learning_rate_schedule == "cosine":
        if last_epoch > 0:
            for pg in optimizer.param_groups:
                pg.setdefault("initial_lr", cfg.train.starting_learning_rate)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.train.num_train_steps,
            eta_min=cfg.train.final_learning_rate,
            last_epoch=last_epoch if last_epoch > 0 else -1,
        )
    else:
        scheduler = None

    # Define the metrics
    metrics = defaultdict(list)
    train_metrics = defaultdict(list)

    # Start the training loop
    step_offset = last_epoch + 1 if last_epoch > 0 else 0

    pbar = tqdm(
        enumerate(train_loader, start=step_offset),
        desc="Training",
        disable=not dist.is_main_process(),
        initial=step_offset,
    )

    if dist.is_enabled():
        torch.distributed.barrier()

    start_train_time = time.time()
    for global_step, batch in pbar:
        batch_start_time = time.time()
        # Make the model in train mode
        digress.diffuser.train()

        # zero grad before forward pass
        optimizer.zero_grad()

        # Reset memory tracker for this step
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        # Move the batch to the correct device
        batch = [v.to(device) if isinstance(v, torch.Tensor) else v for v in batch]

        # Do the forward and loss computation
        try:
            p_nodes, p_edges, loss = digress.forward_backward(*batch)
            skip = torch.zeros(1, device=device)

        except LinAlgError:
            print("LinAlgError in forward or digress_loss")
            p_nodes, p_edges, loss = None, None, None
            skip = torch.ones(1, device=device)

        if dist.is_enabled():
            torch.distributed.all_reduce(skip, op=torch.distributed.ReduceOp.MAX)

        if skip.item():
            optimizer.zero_grad()
            continue

        # All-reduce loss scalar so the logged value is the global average (DDP only)
        loss_for_log = loss
        if loss is not None and cfg.train.get("distributed") == "data":
            loss_for_log = loss.detach().clone()
            torch.distributed.all_reduce(
                loss_for_log, op=torch.distributed.ReduceOp.AVG
            )

        # If loss is None (on non-last ranks in PP), we skip logging
        if loss is not None:
            nodes, edges, mask, *_ = batch
            epoch_metrics = compute_training_metrics(
                p_nodes, p_edges, nodes, edges, mask, digress.valid_elements
            )

            for k, v in epoch_metrics.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                train_metrics[k].append(v)

            # Update the progress bar
            lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=loss_for_log.item(), lr=lr, **epoch_metrics)

            # Log training metrics to wandb
            train_log = {
                "train/loss": loss_for_log.item(),
                "train/learning_rate": lr,
                "train/epoch_time": time.time() - batch_start_time,
                "train/total_time": time.time() - start_train_time,
                "step": global_step,
            }

            for k, v in epoch_metrics.items():
                train_log[f"train/{k}"] = v

            if dist.is_main_process():
                wandb.log(train_log, step=global_step)

        # Update the parameters
        optimizer.step()

        # Step the scheduler
        if scheduler is not None:
            scheduler.step()

        # Start the evaluation
        if global_step % cfg.valid.num_eval_steps == 0:
            # Make the model in eval mode
            digress.diffuser.eval()

            # Do the evaluation — all ranks must participate for pipeline parallelism
            with torch.no_grad():
                eval_metrics = do_eval(
                    digress, valid_loader, device, cfg.valid.batch_size
                )

                # If DDP then we make each rank generates a part of the samples
                if cfg.train.get("distributed") == "data":
                    world_size = dist.get_global_size()
                    samples_per_rank = max(1, cfg.valid.num_samples // world_size)
                else:
                    samples_per_rank = cfg.valid.num_samples

                validity, validity_digress, valid_smiles = generate_samples(
                    digress,
                    samples_per_rank,
                    train_dataset.num_atoms_dist,
                    cfg.valid.batch_size,
                )

                # In DDP mode, gather results from all ranks.
                # all_reduce / all_gather_object act as implicit barriers, so no
                # explicit barrier() is needed.
                if cfg.train.get("distributed") == "data":
                    validity_tensor = torch.tensor(
                        [validity, validity_digress], device=device
                    )
                    torch.distributed.all_reduce(
                        validity_tensor, op=torch.distributed.ReduceOp.SUM
                    )
                    validity_tensor /= world_size
                    validity = validity_tensor[0].item()
                    validity_digress = validity_tensor[1].item()

                    all_valid_smiles = [None] * world_size
                    torch.distributed.all_gather_object(all_valid_smiles, valid_smiles)
                    if dist.is_main_process():
                        valid_smiles = [
                            s for per_rank in all_valid_smiles for s in per_rank
                        ]

                if eval_metrics:
                    eval_metrics["gen_validity"] = validity
                    eval_metrics["gen_validity_digress"] = validity_digress

                    for k, v in eval_metrics.items():
                        metrics[k].append(v)

                    if dist.is_main_process():
                        print(f"Evaluation Metrics: {eval_metrics}")

                        # Log evaluation metrics to wandb
                        eval_log = {"step": global_step}
                        for k, v in eval_metrics.items():
                            eval_log[f"eval/{k}"] = v

                        if valid_smiles:
                            smiles_table = wandb.Table(columns=["SMILES"])
                            for s in valid_smiles:
                                smiles_table.add_data(s)
                            eval_log["eval/generated_smiles"] = smiles_table

                        wandb.log(eval_log, step=global_step)

                        # Save the valid smiles
                        with open(
                            os.path.join(
                                cfg.train.save_path, f"valid_smiles_{global_step}.txt"
                            ),
                            "w",
                        ) as f:
                            for s in valid_smiles:
                                f.write(f"{s}\n")

        # Save the model
        if (global_step % cfg.train.save_steps == 0) & (global_step > 0):
            if cfg.train.get("distributed") == "pipeline":
                save_checkpoint(
                    digress.diffuser,
                    optimizer,
                    os.path.join(cfg.train.save_path, f"checkpoint_{global_step}"),
                )

            else:
                if dist.is_main_process():
                    torch.save(
                        {
                            "model": digress.diffuser.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        },
                        os.path.join(cfg.train.save_path, f"{global_step}.pt"),
                    )

        # Stop the training when the desired number of steps has been reached
        if global_step >= cfg.train.num_train_steps:
            break

    # Save the final model
    if cfg.train.get("distributed") == "pipeline":
        save_checkpoint(
            digress.diffuser,
            optimizer,
            os.path.join(cfg.train.save_path, "final"),
        )

    else:
        if dist.is_main_process():
            torch.save(
                {
                    "model": digress.diffuser.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                os.path.join(cfg.train.save_path, "final.pt"),
            )

    # Save the metrics (main rank only to avoid concurrent write corruption)
    if dist.is_main_process():
        with open(os.path.join(cfg.train.save_path, "val_metrics.json"), "w") as f:
            json.dump(metrics, f)

        with open(os.path.join(cfg.train.save_path, "train_metrics.json"), "w") as f:
            json.dump(train_metrics, f)


def main():
    args = get_args()
    cfg = setup(args)

    if cfg.train.get("distributed") == "pipeline":
        main_rank = "last"

    else:
        main_rank = "first"

    dist.enable(overwrite=True, main_rank=main_rank)
    train(cfg)


if __name__ == "__main__":
    main()
