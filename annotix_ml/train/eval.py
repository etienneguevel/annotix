from collections import defaultdict

import torch
from rdkit.RDLogger import DisableLog
from torch.linalg import LinAlgError
from torch.utils.data import DataLoader
from tqdm import tqdm

import annotix_ml.distributed as dist
from annotix_ml.data.atoms_data import TYPE_EDGES
from annotix_ml.data.data_utils import (
    batch_graph_to_smiles,
    batch_graph_to_smiles_digress,
)
from annotix_ml.graphtransf.arch import DigressMetaArch, Spec2MolMetaArch
from annotix_ml.graphtransf.math.metrics import (
    compute_MCES_distance,
    compute_metrics,
    compute_tanimoto_similarity,
    compute_validity,
)


def do_eval(
    model: DigressMetaArch,
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

        # Unpack the elements
        nodes, edges, mask, *args = batch

        # If last batch, then size will be < to the one planned by schedule
        # To avoid error -> skip it when there is a schedule (ie pp)
        if model.eval_schedule:
            if nodes.shape[0] != expected_bs:
                continue

        # Noise the elements
        nodes_noised, edges_noised, sampled_t = model.noiser(nodes, edges, mask)

        # Compute the predictions
        try:
            p_nodes, p_edges = model.forward(
                nodes_noised, edges_noised, mask, sampled_t, *args
            )  # (bs, n, n_atoms), (bs, n, n, n_edges)
        except LinAlgError:
            continue

        # Non-last ranks in pipeline parallelism get None outputs;
        # they must keep looping so every rank calls schedule.step().
        if p_nodes is None or p_edges is None:
            continue

        # Make the prediction graph
        N_ = torch.nn.functional.one_hot(
            p_nodes.argmax(-1), num_classes=p_nodes.shape[-1]
        )  # (bs, n, n_atoms)
        E_ = torch.nn.functional.one_hot(
            p_edges.argmax(-1), num_classes=p_edges.shape[-1]
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
            p_nodes,
            p_edges,
            nodes,
            edges,
            mask,
        )

        # Compute accuracy per node type
        target_nodes = nodes.argmax(-1)
        pred_nodes = p_nodes.argmax(-1)
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
        pred_edges = p_edges.argmax(-1)
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


def allreduce_eval_metrics(
    eval_metrics: dict[str, float],
) -> dict[str, float]:
    """Average eval metrics across all distributed ranks via all_gather_object."""
    if not dist.is_enabled() or not eval_metrics:
        return eval_metrics

    world_size = dist.get_global_size()
    all_metrics = [None] * world_size
    torch.distributed.all_gather_object(all_metrics, eval_metrics)

    # Merge: for each key, average over ranks that reported it
    all_keys: set[str] = set()
    for m in all_metrics:
        all_keys.update(m.keys())

    result = {}
    for k in all_keys:
        values = [m[k] for m in all_metrics if k in m]
        result[k] = sum(values) / len(values)

    return result


def allreduce_gen_metrics(gen_metrics: dict[str, list]) -> dict[str, list]:
    """Gather per-sample gen_metrics lists from all DDP ranks and concatenate."""
    if not dist.is_enabled() or not gen_metrics:
        return gen_metrics

    world_size = dist.get_global_size()
    all_metrics = [None] * world_size
    torch.distributed.all_gather_object(all_metrics, gen_metrics)

    result: dict[str, list] = {}
    for k in all_metrics[0].keys():
        result[k] = [item for rank_dict in all_metrics for item in rank_dict[k]]
    return result


def generate_samples(
    model: DigressMetaArch,
    num_samples: int,
    num_nodes_dist: torch.Tensor,
    batch_size: int,
) -> dict[str, list]:
    # Generate the graphs
    batch_size = min(num_samples, batch_size)
    print(f"Generating {num_samples} graphs in batches of {batch_size}...")

    # Init the lists
    all_gen_smiles = []
    all_gen_smiles_digress = []
    samples_generated = 0

    while samples_generated < num_samples:
        # Sample from the distribution
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

            # Convert to smiles
            gen_smiles = batch_graph_to_smiles(
                gen_N, gen_E, gen_mask, model.valid_elements
            )
            all_gen_smiles.extend(gen_smiles)

            # Convert to smiles with Digress method
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

    return {
        "all_gen_smiles": all_gen_smiles,
        "all_gen_smiles_digress": all_gen_smiles_digress,
        "validity": [1.0 if s else 0.0 for s in all_gen_smiles],
        "validity_digress": [1.0 if s else 0.0 for s in all_gen_smiles_digress],
    }


def generate_samples_from_spec(
    model: Spec2MolMetaArch,
    num_samples: int,
    eval_dataloader: DataLoader,
):
    # Init the lists
    metrics = defaultdict(list)
    samples_generated = 0

    # Make the eval loader as iterable
    iter_loader = iter(eval_dataloader)

    while samples_generated < num_samples:
        try:
            batch = next(iter_loader)
        except StopIteration:
            print("Dataloader exhausted before reaching num_samples.")
            break
        # Unpack the batch
        batch = [
            k.to(model.device) if isinstance(k, torch.Tensor) else k for k in batch
        ]
        nodes, edges, mask, *spectra_args = batch
        bs = nodes.shape[0]

        # Generate the samples from the spectra_args
        n = mask.sum(-1)  # Generate molecules with the same number of nodes as original

        try:
            gen_N, gen_E, gen_mask = model.generate(
                num_samples=n,
                max_nodes=n.max(),
                progress_bar=True,
                other_args=spectra_args,
            )

            # Convert to smiles
            gen_smiles = batch_graph_to_smiles(
                gen_N, gen_E, gen_mask, model.valid_elements
            )
            metrics["all_gen_smiles"].extend(gen_smiles)

            # Convert to smiles with Digress method
            gen_smiles_digress = batch_graph_to_smiles_digress(
                gen_N, gen_E, gen_mask, model.valid_elements
            )
            metrics["all_gen_smiles_digress"].extend(gen_smiles_digress)

        except LinAlgError:
            print("LinAlgError during generation batch, skipping batch...")
            continue

        # Compute the metrics between the generated samples and original molecules
        true_smiles = batch_graph_to_smiles(nodes, edges, mask, model.valid_elements)
        tan_sim = compute_tanimoto_similarity(gen_smiles, true_smiles)
        mces = compute_MCES_distance(gen_smiles, true_smiles)

        metrics["true_smiles"].extend(true_smiles)
        metrics["validity"].extend(compute_validity(gen_smiles))
        metrics["validity_digress"].extend(compute_validity(gen_smiles_digress))
        metrics["tan_sim"].extend(tan_sim)
        metrics["mces"].extend(mces)

        # Increase the counter of generated samples
        samples_generated += bs

    return metrics
