import contextlib
import json
import os
import time
from argparse import ArgumentParser
from collections import defaultdict

import torch
import wandb
from omegaconf import OmegaConf
from torch.linalg import LinAlgError
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import annotix_ml.distributed as dist
from annotix_ml.graphtransf.arch import DigressMetaArch
from annotix_ml.distributed.pipeline_parallelism import save_checkpoint
from annotix_ml.data.samplers import InfiniteSampler
from annotix_ml.graphtransf.math.metrics import (
    compute_training_metrics,
)
from annotix_ml.train.eval import (
    allreduce_eval_metrics,
    allreduce_gen_metrics,
    compute_gen_retrieval_metrics,
    do_eval,
    generate_samples,
    generate_samples_from_fp,
)
from annotix_ml.train.setup import setup, setup_model_mode
from annotix_ml.train.log_utils import create_gen_samples_table


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
    parser.add_argument(
        "--main-rank-only-generation",
        action="store_true",
        default=False,
    )
    parser.add_argument("--pretrain-path", type=str, required=False)

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
    train_dataset, valid_dataset, collate_fn = setup_model_mode(cfg, device)

    if (max_eval := cfg.valid.get("max_eval_samples")) is not None:
        rng = torch.Generator().manual_seed(cfg.train.seed)
        indices = torch.randperm(len(valid_dataset), generator=rng)[:max_eval].tolist()
        valid_dataset = Subset(valid_dataset, indices)

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
        digress = DigressMetaArch.load_pretrained(
            cfg,
            device,
            checkpoint_path,
            train_dataset.valid_elements,
            train_dataset.nodes_distribution,
            train_dataset.edges_distribution,
            train_dataset.max_weight,
        )

    elif (pretrain_path := cfg.train.pretrain_path) is not None:
        with open(os.path.join(pretrain_path, "dataset_meta.json")) as f:
            dataset_meta = json.load(f)

        model_path = os.path.join(pretrain_path, "pretrained_model.pt")

        # Check if the atoms used for pretrained model and this dataset are the same
        if set(train_dataset.valid_elements) == set(dataset_meta.get("valid_elements")):
            digress = DigressMetaArch.load_pretrained(
                cfg,
                device,
                model_path,
                train_dataset.valid_elements,
                train_dataset.nodes_distribution,
                train_dataset.edges_distribution,
                train_dataset.max_weight,
            )

        else:
            pretrain_model = torch.load(
                model_path, weights_only=True, map_location=device
            )
            weights = (
                pretrain_model.get("checkpoint").get("model")
                if "checkpoint" in pretrain_model.keys()
                else pretrain_model.get("model")
            )
            weights = {k: v for k, v in weights.items() if "EmbeddingNodes" not in k}

            digress = DigressMetaArch.init_from_cfg(
                cfg,
                device,
                train_dataset.valid_elements,
                train_dataset.nodes_distribution,
                train_dataset.edges_distribution,
                max_weight=train_dataset.max_weight,
            )
            digress.load_checkpoint_state_dict({"model": weights}, strict=False)

    else:
        # Define the model
        print("No checkpoints found, init from zero.")
        digress = DigressMetaArch.init_from_cfg(
            cfg,
            device,
            train_dataset.valid_elements,
            train_dataset.nodes_distribution,
            train_dataset.edges_distribution,
            max_weight=train_dataset.max_weight,
        )

    # Compile the diffuser for faster training (skip for pipeline parallelism)
    if cfg.train.get("compile", False) and cfg.train.get("distributed") != "pipeline":
        digress.diffuser = torch.compile(digress.diffuser)
        if digress.fp_projector is not None:
            digress.fp_projector = torch.compile(digress.fp_projector)

    samples_seen_offset = 0
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

        # Recover total_samples_seen offset for resumed runs
        if last_epoch > 0:
            summary_val = wandb.run.summary.get("train/total_samples_seen")
            if summary_val is not None:
                samples_seen_offset = int(summary_val)
            else:
                metadata_path = os.path.join(save_path, "training_metadata.json")
                if os.path.isfile(metadata_path):
                    with open(metadata_path) as f:
                        samples_seen_offset = json.load(f).get("total_samples_seen", 0)

    # Compute gradient accumulation steps
    is_ddp = dist.is_enabled() and cfg.train.get("distributed") == "data"

    if cfg.train.get("distributed") == "pipeline":
        accum_steps = 1
    else:
        effective_per_step = cfg.train.mini_batch_size * data_size
        accum_steps = max(1, cfg.train.global_batch_size // effective_per_step)
        actual_global_batch = effective_per_step * accum_steps
        if actual_global_batch != cfg.train.global_batch_size:
            print(
                f"Warning: global_batch_size={cfg.train.global_batch_size} is not evenly "
                f"divisible by mini_batch_size * world_size "
                f"({cfg.train.mini_batch_size} * {data_size}). "
                f"Using effective global batch size of {actual_global_batch}."
            )
        print(
            f"Gradient accumulation: {accum_steps} steps "
            f"(mini_batch={cfg.train.mini_batch_size}, world_size={data_size}, "
            f"effective_global_batch={actual_global_batch})"
        )

    # Make the DataLoaders
    sample_count = len(train_dataset)
    shuffle = True
    seed = cfg.train.seed
    advance = (
        ((last_epoch + 1) * accum_steps * cfg.train.mini_batch_size)
        % len(train_dataset)
        if (last_epoch > 0 and not shuffle)
        else 0
    )

    # If the training is pipeline dist -> train and valid bs need to be =
    # TODO : find a way to rm that dependency
    if cfg.train.get("distributed") == "pipeline":
        cfg.valid.batch_size = cfg.train.mini_batch_size
        data_rank = 0
        data_size = 1
        print(
            f"Pipeline Parallelism detected: setting data_rank={data_rank}, data_size={data_size} to synchronize input/targets"
        )

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
        batch_size=cfg.train.mini_batch_size,
        sampler=sampler_type,
        collate_fn=collate_fn,
        num_workers=cfg.train.num_workers,
    )
    if cfg.train.get("distributed") == "data":
        valid_sampler = DistributedSampler(
            valid_dataset,
            num_replicas=data_size,
            rank=data_rank,
            shuffle=True,
            seed=12,
        )
    else:
        valid_sampler = None

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.valid.batch_size,
        collate_fn=collate_fn,
        sampler=valid_sampler,
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
        digress.trainable_parameters(),
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

    # Save dataset metadata once so the model can be reloaded without the dataset
    if dist.is_main_process():
        dataset_meta_path = os.path.join(save_path, "dataset_meta.json")
        if not os.path.isfile(dataset_meta_path):
            with open(dataset_meta_path, "w") as f:
                json.dump(
                    {
                        "valid_elements": train_dataset.valid_elements,
                        "nodes_distribution": train_dataset.nodes_distribution.tolist(),
                        "edges_distribution": train_dataset.edges_distribution.tolist(),
                        "max_weight": train_dataset.max_weight,
                        "num_atoms_dist": train_dataset.num_atoms_dist.tolist(),
                    },
                    f,
                )

    # Start the training loop
    step_offset = last_epoch + 1 if last_epoch > 0 else 0

    optimizer.zero_grad()
    train_iter = iter(train_loader)

    pbar = tqdm(
        range(step_offset, cfg.train.num_train_steps),
        desc="Training",
        disable=not dist.is_main_process(),
    )

    if dist.is_enabled():
        torch.distributed.barrier()

    start_train_time = time.time()
    for global_step in pbar:
        batch_start_time = time.time()
        digress.diffuser.train()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        # --- Gradient accumulation inner loop ---
        skip_window = False
        losses_for_log = []
        accumulated_metrics = defaultdict(list)

        for accum_i in range(accum_steps):
            batch = next(train_iter)
            batch = [v.to(device) if isinstance(v, torch.Tensor) else v for v in batch]

            is_last_accum = accum_i == accum_steps - 1
            sync_ctx = (
                digress.diffuser.no_sync()
                if is_ddp and not is_last_accum
                else contextlib.nullcontext()
            )

            try:
                with sync_ctx:
                    p_nodes, p_edges, loss = digress.forward_backward(
                        *batch, loss_scale=1.0 / accum_steps
                    )
                skip = torch.zeros(1, device=device)

            except LinAlgError:
                print("LinAlgError in forward or digress_loss")
                p_nodes, p_edges, loss = None, None, None
                skip = torch.ones(1, device=device)

            if dist.is_enabled():
                torch.distributed.all_reduce(skip, op=torch.distributed.ReduceOp.MAX)

            if skip.item():
                skip_window = True
                break

            if loss is not None:
                losses_for_log.append(loss.detach())
                nodes, edges, mask, *_ = batch
                step_metrics = compute_training_metrics(
                    p_nodes, p_edges, nodes, edges, mask, digress.valid_elements
                )
                for k, v in step_metrics.items():
                    accumulated_metrics[k].append(
                        v.item() if isinstance(v, torch.Tensor) else v
                    )

        if skip_window:
            optimizer.zero_grad()
            continue

        # --- Post-accumulation: sync extra gradients (DDP only) ---
        if dist.is_enabled() and cfg.train.get("distributed") == "data":
            digress.sync_extra_gradients()

        total_samples_seen = (
            samples_seen_offset
            + (global_step - step_offset) * cfg.train.global_batch_size
        )

        # --- Logging ---
        if losses_for_log:
            avg_loss = sum(losses_for_log) / len(losses_for_log)
            loss_for_log = avg_loss
            if cfg.train.get("distributed") == "data":
                loss_for_log = avg_loss.clone()
                torch.distributed.all_reduce(
                    loss_for_log, op=torch.distributed.ReduceOp.AVG
                )

            epoch_metrics = {k: sum(v) / len(v) for k, v in accumulated_metrics.items()}

            for k, v in epoch_metrics.items():
                train_metrics[k].append(v)

            lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=loss_for_log.item(), lr=lr, **epoch_metrics)

            if dist.is_main_process():
                train_log = {
                    "train/loss": loss_for_log.item(),
                    "train/learning_rate": lr,
                    "train/epoch_time": time.time() - batch_start_time,
                    "train/total_time": time.time() - start_train_time,
                    "train/total_samples_seen": total_samples_seen,
                }
                for k, v in epoch_metrics.items():
                    train_log[f"train/{k}"] = v
                wandb.log(train_log, step=global_step)

        # --- Optimizer step ---
        digress.scaler.step(optimizer)
        digress.scaler.update()
        optimizer.zero_grad()

        # Step the scheduler
        if scheduler is not None:
            scheduler.step()

        # Save the model
        if (global_step % cfg.train.save_steps == 0) & (global_step > 0):
            if cfg.train.get("distributed") == "pipeline":
                save_checkpoint(
                    digress.diffuser,
                    optimizer,
                    os.path.join(cfg.train.save_path, f"checkpoint_{global_step}"),
                )
                if dist.is_main_process():
                    metadata_path = os.path.join(
                        cfg.train.save_path, "training_metadata.json"
                    )
                    with open(metadata_path, "w") as f:
                        json.dump({"total_samples_seen": total_samples_seen}, f)

            else:
                if dist.is_main_process():
                    torch.save(
                        {
                            "checkpoint": digress.checkpoint_state_dict(),
                            "optimizer": optimizer.state_dict(),
                        },
                        os.path.join(cfg.train.save_path, f"{global_step}.pt"),
                    )
                    metadata_path = os.path.join(
                        cfg.train.save_path, "training_metadata.json"
                    )
                    with open(metadata_path, "w") as f:
                        json.dump({"total_samples_seen": total_samples_seen}, f)

        # Start the evaluation
        if global_step % cfg.valid.num_eval_steps == 0:
            # Make the model in eval mode
            digress.diffuser.eval()

            # Do the evaluation — all ranks must participate for pipeline parallelism
            with torch.no_grad():
                eval_metrics = do_eval(
                    digress, valid_loader, device, cfg.valid.batch_size
                )

                if cfg.train.get("distributed") == "data":
                    torch.distributed.barrier()
                    eval_metrics = allreduce_eval_metrics(eval_metrics)

                main_rank_only_gen = cfg.valid.get("main_rank_only_generation", False)

                # If DDP then we make each rank generates a part of the samples
                if cfg.train.get("distributed") == "data" and not main_rank_only_gen:
                    world_size = dist.get_global_size()
                    samples_per_rank = max(1, cfg.valid.num_samples // world_size)

                else:
                    samples_per_rank = cfg.valid.num_samples

                should_generate = not main_rank_only_gen or dist.is_main_process()

                if should_generate:
                    if "spectra_fingerprint" in cfg.model.get("extra_features"):
                        gen_metrics = generate_samples_from_fp(
                            digress, samples_per_rank, valid_loader
                        )
                    else:
                        gen_metrics = generate_samples(
                            digress,
                            samples_per_rank,
                            train_dataset.num_atoms_dist,
                            cfg.valid.batch_size,
                        )
                else:
                    gen_metrics = {}

                if cfg.train.get("distributed") == "data" and not main_rank_only_gen:
                    torch.distributed.barrier()
                    gen_metrics = allreduce_gen_metrics(gen_metrics)

                # Compute expensive retrieval metrics (MCES, Tanimoto) only on
                # main rank after allreduce, so other ranks aren't blocked.
                if dist.is_main_process() and "spectra_fingerprint" in cfg.model.get(
                    "extra_features"
                ):
                    gen_metrics = compute_gen_retrieval_metrics(gen_metrics)

                validity = (
                    sum(gen_metrics["validity"]) / len(gen_metrics["validity"])
                    if gen_metrics.get("validity")
                    else 0.0
                )
                validity_digress = (
                    sum(gen_metrics["validity_digress"])
                    / len(gen_metrics["validity_digress"])
                    if gen_metrics.get("validity_digress")
                    else 0.0
                )
                valid_smiles = [s for s in gen_metrics.get("all_gen_smiles", []) if s]

                if eval_metrics:
                    eval_metrics["gen_validity"] = validity
                    eval_metrics["gen_validity_digress"] = validity_digress

                    for k, v in eval_metrics.items():
                        metrics[k].append(v)

                    if dist.is_main_process():
                        print(f"Evaluation Metrics: {eval_metrics}")

                        # Log evaluation metrics to wandb
                        eval_log = {}
                        for k, v in eval_metrics.items():
                            eval_log[f"eval/{k}"] = v

                        if (
                            "spectra_fingerprint" in cfg.model.extra_features
                        ) and gen_metrics:
                            eval_log["generated_samples"] = create_gen_samples_table(
                                gen_metrics
                            )
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

            # Barrier so all ranks are synchronized before the next training step.
            # Prevents NCCL timeouts when eval/generation takes asymmetric time.
            if cfg.train.get("distributed") == "data":
                torch.distributed.barrier()

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
        if dist.is_main_process():
            metadata_path = os.path.join(cfg.train.save_path, "training_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump({"total_samples_seen": total_samples_seen}, f)

    else:
        if dist.is_main_process():
            torch.save(
                {
                    "checkpoint": digress.checkpoint_state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                os.path.join(cfg.train.save_path, "final.pt"),
            )
            metadata_path = os.path.join(cfg.train.save_path, "training_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump({"total_samples_seen": total_samples_seen}, f)

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

    dist.enable(
        overwrite=True,
        main_rank=main_rank,
        allow_nccl_timeout=True,
        timeout_minutes=cfg.train.get("nccl_timeout_minutes", 30),
    )
    train(cfg)

    if dist.is_enabled():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
