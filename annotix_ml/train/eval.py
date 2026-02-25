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
from annotix_ml.graphtransf.arch import DigressMetaArch
from annotix_ml.graphtransf.math.metrics import compute_metrics


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


def generate_samples(
    model: DigressMetaArch,
    num_samples: int,
    num_nodes_dist: torch.Tensor,
    batch_size: int,
) -> tuple[float, float, list[str]]:
    # Generate the graphs
    batch_size = min(num_samples, batch_size)

    print(f"Generating {num_samples} graphs in batches of {batch_size}...")
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

    valid_smiles = [s for s in all_gen_smiles if s]
    valid_smiles_digress = [s for s in all_gen_smiles_digress if s]

    # Compute the validity
    validity = len(valid_smiles) / len(all_gen_smiles) if all_gen_smiles else 0
    validity_digress = (
        len(valid_smiles_digress) / len(all_gen_smiles_digress)
        if all_gen_smiles_digress
        else 0
    )

    return validity, validity_digress, valid_smiles
