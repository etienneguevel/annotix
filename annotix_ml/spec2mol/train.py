import json
import os
import shutil
from collections import defaultdict

import pandas as pd
import torch
import wandb
from rdkit.RDLogger import DisableLog  # pyright: ignore[reportAttributeAccessIssue]
from omegaconf import OmegaConf
from torch.linalg import LinAlgError
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

import annotix_ml.distributed as dist
from annotix_ml.distributed import enable
from annotix_ml.distributed.pipeline_parallelism import save_checkpoint
from annotix_ml.graphtransf.data.atoms_data import TYPE_EDGES
from annotix_ml.graphtransf.data.data_utils import (
    batch_graph_to_smiles,
    batch_graph_to_smiles_digress,
    graph_to_smiles_digress,
)
from annotix_ml.graphtransf.data.samplers import InfiniteSampler
from annotix_ml.graphtransf.math.metrics import (
    compute_metrics,
    compute_training_metrics,
)
from annotix_ml.graphtransf.train.train import get_args
from annotix_ml.graphtransf.train.setup import setup
from annotix_ml.spec2mol.data.datacollator import graph_spec_collate_fn
from annotix_ml.spec2mol.data.dataset import GraphSpecDataset
from annotix_ml.spec2mol.spec2mol_meta_arch import Spec2MolMetaArch


def make_datasets(cfg):
    """Create train/valid GraphSpecDataset from config."""
    data = pd.read_csv(
        cfg.dataset.data_path,
        sep="\t" if str(cfg.dataset.data_path).endswith(".tsv") else ",",
    )

    split_column = cfg.dataset.split_column
    val_tag = cfg.dataset.val_tag

    train_df = data[data[split_column] != val_tag].reset_index(drop=True)
    valid_df = data[data[split_column] == val_tag].reset_index(drop=True)

    cache_path = cfg.dataset.get("cache_path")
    train_cache = f"{cache_path}_train.pt" if cache_path else None
    valid_cache = f"{cache_path}_valid.pt" if cache_path else None

    common_kwargs = dict(
        spec_folder=cfg.dataset.spec_folder,
        subform_folder=cfg.dataset.subform_folder,
        smile_column=cfg.dataset.smile_column,
        sanitizer=graph_to_smiles_digress,
    )

    train_dataset = GraphSpecDataset(
        data=train_df,
        cache_path=train_cache,
        save_cache=dist.is_main_process(),
        **common_kwargs,
    )
    valid_elements = train_dataset.valid_elements
    valid_dataset = GraphSpecDataset(
        data=valid_df,
        valid_elements=valid_elements,
        cache_path=valid_cache,
        save_cache=dist.is_main_process(),
        **common_kwargs,
    )

    return train_dataset, valid_dataset


def do_eval(
    model: Spec2MolMetaArch,
    eval_loader: DataLoader,
    device: torch.device,
    expected_bs: int | None = None,
) -> dict[str, float]:
    DisableLog("rdApp.*")
    metrics = defaultdict(list)
    for batch in tqdm(
        eval_loader, desc="evaluation", disable=not dist.is_main_process()
    ):
        # Move the elements to the device of interest
        batch = [v.to(device) if isinstance(v, torch.Tensor) else v for v in batch]

        # Unpack the 10-element tuple
        (
            nodes,
            edges,
            mask,
            num_peaks,
            types,
            instruments,
            ion_vec,
            form_vec,
            intens,
            smiles,
        ) = batch

        # If last batch, then size will be < to the one planned by schedule
        # To avoid error -> skip it when there is a schedule (ie pp)
        if model.eval_schedule:
            if nodes.shape[0] != expected_bs:
                continue

        # Noise the elements
        nodes_noised, edges_noised, sampled_t = model.noiser(nodes, edges, mask)

        # Compute the predictions
        pN, pE = model.forward(
            nodes_noised,
            edges_noised,
            mask,
            sampled_t,
            num_peaks=num_peaks,
            types=types,
            instruments=instruments,
            ion_vec=ion_vec,
            form_vec=form_vec,
            intens=intens,
        )

        # Non-last ranks in pipeline parallelism get None outputs;
        # they must keep looping so every rank calls schedule.step().
        if pN is None or pE is None:
            continue

        # Make the prediction graph
        N_ = torch.nn.functional.one_hot(pN.argmax(-1), num_classes=pN.shape[-1])
        E_ = torch.nn.functional.one_hot(pE.argmax(-1), num_classes=pE.shape[-1])

        # Convert the graph to smiles
        true_smiles = batch_graph_to_smiles(nodes, edges, mask, model.valid_elements)
        pred_smiles = batch_graph_to_smiles(N_, E_, mask, model.valid_elements)

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
    model: Spec2MolMetaArch,
    num_samples: int,
    num_nodes_dist: torch.Tensor,
    batch_size: int,
) -> tuple[float, float, list[str]]:
    print(f"Generating {num_samples} graphs in batches of {batch_size}...")
    all_gen_smiles = []
    all_gen_smiles_digress = []
    samples_generated = 0

    while samples_generated < num_samples:
        n = (
            num_nodes_dist.unsqueeze(0)
            .expand((batch_size, -1))
            .multinomial(1)
            .squeeze(-1)
            + 1
        )

        try:
            gen_N, gen_E, gen_mask = model.generate(
                num_samples=n, max_nodes=n.max(), progress_bar=True
            )

            gen_smiles = batch_graph_to_smiles(
                gen_N, gen_E, gen_mask, model.valid_elements
            )
            all_gen_smiles.extend(gen_smiles)

            gen_smiles_digress = batch_graph_to_smiles_digress(
                gen_N, gen_E, gen_mask, model.valid_elements
            )
            all_gen_smiles_digress.extend(gen_smiles_digress)

        except LinAlgError:
            print("LinAlgError during generation batch, skipping batch...")
            continue

        samples_generated += batch_size

    # Trim to exactly num_samples to avoid overshoot bias in validity metrics.
    all_gen_smiles = all_gen_smiles[:num_samples]
    all_gen_smiles_digress = all_gen_smiles_digress[:num_samples]

    valid_smiles = [s for s in all_gen_smiles if s]
    valid_smiles_digress = [s for s in all_gen_smiles_digress if s]

    validity = len(valid_smiles) / len(all_gen_smiles) if all_gen_smiles else 0
    validity_digress = (
        len(valid_smiles_digress) / len(all_gen_smiles_digress)
        if all_gen_smiles_digress
        else 0
    )

    return validity, validity_digress, valid_smiles


def train(cfg):
    # Initialize wandb
    if dist.is_main_process():
        print(f"Logging with main rank as : {dist._MAIN_RANK}")

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

    train_dataset, valid_dataset = make_datasets(cfg)

    # Determine distributed data rank/size
    data_rank = dist.get_global_rank()
    data_size = dist.get_global_size()

    if dist.is_enabled():
        if cfg.train.get("distributed") == "pipeline":
            data_rank = 0
            data_size = 1
            print(
                f"Pipeline Parallelism detected: setting data_rank={data_rank}, data_size={data_size} to synchronize input/targets"
            )

    # If the training is pipeline dist -> train and valid bs need to be =
    if cfg.train.get("distributed") == "pipeline":
        cfg.valid.batch_size = cfg.train.batch_size

    collate_fn = graph_spec_collate_fn

    # Make the DataLoaders
    sample_count = len(train_dataset)
    sampler_type = InfiniteSampler(
        sample_count=sample_count,
        shuffle=True,
        seed=cfg.train.seed,
        start=data_rank,
        step=data_size,
        advance=0,
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

    # Define the model
    spec2mol = Spec2MolMetaArch.init_from_cfg(
        cfg,
        device,
        train_dataset.valid_elements,
        train_dataset.nodes_distribution,
        train_dataset.edges_distribution,
        max_weight=train_dataset.max_weight,
    )
    spec2mol.valid_elements = train_dataset.valid_elements

    # Log model dimensions
    print("\n" + "=" * 50)
    print("MODEL DIMENSIONS")
    print("=" * 50)
    print(f"Hidden dimension (d): {cfg.model.d}")
    print(f"Edge dimension (de): {cfg.model.de}")
    print(f"Global features dimension (dy): {cfg.model.dy}")
    print(f"Number of attention heads: {cfg.model.n_heads}")
    print(f"Number of layers: {cfg.model.n_layers}")
    print(f"Number of atom types (natoms): {spec2mol.natoms}")
    print(f"Number of bond types (nbonds): {spec2mol.nbonds}")
    if hasattr(cfg.model, "extra_features") and cfg.model.extra_features:
        print(f"Extra features: {cfg.model.extra_features}")
    print("=" * 50 + "\n")

    # Print the diffuser model architecture
    num_param = sum([p.numel() for p in spec2mol.diffuser.parameters()])
    print("=" * 50)
    print("DIFFUSER MODEL ARCHITECTURE")
    print("=" * 50)
    print(f"Num parameters: {num_param / 1e6:.2f}")
    print("=" * 50)
    print(spec2mol.diffuser)
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
    print(f"Number of atoms distribution: {train_dataset.num_atoms_dist.tolist()}")
    print("Number of atoms distribution:")
    num_atoms_dist = train_dataset.num_atoms_dist.tolist()
    bar_width = 30
    max_prob = max(num_atoms_dist) if num_atoms_dist else 1.0
    for idx, prob in enumerate(num_atoms_dist):
        n_atoms = idx + 1
        bar_len = int(prob / max_prob * bar_width)
        bar = "#" * bar_len
        print(f"  {n_atoms:3d} atoms | {bar:<{bar_width}} {prob:.4f}")
    print(f"Max weight in dataset: {train_dataset.max_weight}")
    print("=" * 50 + "\n")
    print(f"Max weight in dataset: {train_dataset.max_weight}")
    print("=" * 50 + "\n")

    # Define the metrics
    metrics = defaultdict(list)
    train_metrics = defaultdict(list)

    # Prepare the save path
    save_path = cfg.train.save_path
    if dist.is_main_process():
        if os.path.isdir(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path)
    if dist.is_enabled():
        torch.distributed.barrier()

    # Setup for distributed training
    if cfg.train.get("distributed") is not None:
        example_batch = next(iter(train_loader))
        spec2mol._setup_distributed(
            cfg.train.distributed, cfg.train.num_microbatches, example_batch
        )

    # Make the optimizer — only diffuser + merge_function params (spectra encoder stays frozen)
    optimizer = torch.optim.AdamW(
        list(spec2mol.diffuser.parameters())
        + list(spec2mol.merge_function.parameters()),
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

    start_train_time = time.time()
    for i, batch in pbar:
        batch_start_time = time.time()
        spec2mol.diffuser.train()

        optimizer.zero_grad()

        # Reset peak memory stats for this step
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        # Move the batch to the correct device
        batch = [v.to(device) if isinstance(v, torch.Tensor) else v for v in batch]
        (
            nodes,
            edges,
            mask,
            num_peaks,
            types,
            instruments,
            ion_vec,
            form_vec,
            intens,
            smiles,
        ) = batch

        # Do the forward and loss computation
        try:
            pN, pE, loss = spec2mol.forward_backward(
                nodes,
                edges,
                mask,
                num_peaks=num_peaks,
                types=types,
                instruments=instruments,
                ion_vec=ion_vec,
                form_vec=form_vec,
                intens=intens,
            )
            skip = torch.zeros(1, device=device)
        except LinAlgError:
            print("LinAlgError in forward or digress_loss")
            pN, pE, loss = None, None, None
            skip = torch.ones(1, device=device)

        if dist.is_enabled():
            torch.distributed.all_reduce(skip, op=torch.distributed.ReduceOp.MAX)
        if skip.item():
            optimizer.zero_grad()
            continue

        # All-reduce loss scalar so the logged value is the global average (DDP only)
        loss_for_log = loss
        if (
            loss is not None
            and dist.is_enabled()
            and spec2mol.train_model is not None
            and spec2mol.train_schedule is None
        ):
            loss_for_log = loss.detach().clone()
            torch.distributed.all_reduce(
                loss_for_log, op=torch.distributed.ReduceOp.AVG
            )

        # If loss is None (on non-last ranks in PP), we skip logging
        if loss is not None:
            epoch_metrics = compute_training_metrics(
                pN, pE, nodes, edges, mask, spec2mol.valid_elements
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
                "step": i,
            }

            for k, v in epoch_metrics.items():
                train_log[f"train/{k}"] = v

            if dist.is_main_process():
                wandb.log(train_log)

        # Update the parameters
        optimizer.step()

        # Step the scheduler
        if scheduler is not None:
            scheduler.step()

        # Start the evaluation
        if i % cfg.valid.num_eval_steps == 0:
            spec2mol.diffuser.eval()

            with torch.no_grad():
                eval_metrics = do_eval(
                    spec2mol, valid_loader, device, cfg.valid.batch_size
                )

                # Pipeline: all ranks must participate; DDP: only main rank needs to run
                if (
                    not dist.is_enabled()
                    or spec2mol.eval_schedule
                    or dist.is_main_process()
                ):
                    validity, validity_digress, valid_smiles = generate_samples(
                        spec2mol,
                        cfg.valid.num_samples,
                        train_dataset.num_atoms_dist,
                        cfg.valid.batch_size,
                    )
                else:
                    validity, validity_digress, valid_smiles = 0.0, 0.0, []

                if eval_metrics:
                    eval_metrics["gen_validity"] = validity
                    eval_metrics["gen_validity_digress"] = validity_digress

                    for k, v in eval_metrics.items():
                        metrics[k].append(v)

                    if dist.is_main_process():
                        print(f"Evaluation Metrics: {eval_metrics}")

                        eval_log = {"step": i}
                        for k, v in eval_metrics.items():
                            eval_log[f"eval/{k}"] = v

                        wandb.log(eval_log)

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
                    spec2mol.diffuser,
                    optimizer,
                    os.path.join(cfg.train.save_path, f"checkpoint_{i}"),
                )
            else:
                if dist.is_main_process():
                    torch.save(
                        {
                            "diffuser": spec2mol.diffuser.state_dict(),
                            "merge_function": spec2mol.merge_function.state_dict(),
                        },
                        os.path.join(cfg.train.save_path, f"{i}.pt"),
                    )

        # Stop the training when the desired number of steps has been reached
        if i >= cfg.train.num_train_steps:
            break

    # Save the final model
    if cfg.train.get("distributed") == "pipeline":
        save_checkpoint(
            spec2mol.diffuser,
            optimizer,
            os.path.join(cfg.train.save_path, "final"),
        )
    else:
        if dist.is_main_process():
            torch.save(
                {
                    "diffuser": spec2mol.diffuser.state_dict(),
                    "merge_function": spec2mol.merge_function.state_dict(),
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

    if cfg.train.distributed == "pipeline":
        main_rank = "last"
    else:
        main_rank = "first"

    enable(overwrite=True, main_rank=main_rank)
    train(cfg)


if __name__ == "__main__":
    main()
