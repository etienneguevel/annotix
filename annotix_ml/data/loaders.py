import pandas as pd
import torch
import torch.nn as nn

from annotix_ml.data.dataset import GraphDatasetFromSMILEs, GraphDatasetWithFP
from annotix_ml.data.spec_dataset import GraphSpecDataset
from annotix_ml.data.data_utils import graph_to_smiles_digress


def make_datasets(
    data_path: str,
    smile_column: str = "smiles",
    split_column: str = "fold",
    val_tag: str | None = "test",
    fingerprint: bool = False,
    verbose: bool = True,
    cache_path: str | None = None,
    save_cache: bool = True,
    max_nodes: int | None = None,
):
    """
    Build train and validation graph datasets from a CSV file of SMILES strings.

    Args:
        data_path (str): Path to the CSV file containing the molecule data.
        smile_column (str): Name of the column containing SMILES strings. Defaults to "smiles".
        split_column (str): Name of the column used for train/val splitting. Defaults to "fold".
        val_tag (str | None): Value in `split_column` that marks validation rows. Defaults to "test".
        fingerprint (bool): If True, use `GraphDatasetWithFP` so each item also returns a
            Morgan fingerprint. If False, use `GraphDatasetFromSMILEs`. Defaults to False.
        verbose (bool): Whether to print progress information. Defaults to True.
        cache_path (str | None): Base path for cache files. Train and validation caches are
            stored at `{cache_path}_train.pt` and `{cache_path}_valid.pt` respectively.
            If None, no caching is performed.
        save_cache (bool): Whether to write the cache files after building. Set to False on
            non-main ranks in distributed training to avoid concurrent writes. Defaults to True.
        max_nodes (int | None): Maximum number of atoms allowed per molecule. Molecules with
            more atoms are filtered out. Defaults to None (no limit).

    Returns:
        tuple[Dataset, Dataset]: A tuple (train_dataset, valid_dataset).
    """
    # Derive per-split cache paths from the base cache_path prefix
    train_cache = f"{cache_path}_train.pt" if cache_path is not None else None
    valid_cache = f"{cache_path}_valid.pt" if cache_path is not None else None

    dataset_type = GraphDatasetWithFP if fingerprint else GraphDatasetFromSMILEs

    # Make the datasets
    train_dataset = dataset_type(
        data=data_path,
        smile_column=smile_column,
        split="train",
        split_column=split_column,
        sanitizer=graph_to_smiles_digress,
        verbose=verbose,
        cache_path=train_cache,
        save_cache=save_cache,
        max_nodes=max_nodes,
    )
    valid_dataset = dataset_type(
        data=data_path,
        smile_column=smile_column,
        split=val_tag,
        split_column=split_column,
        sanitizer=graph_to_smiles_digress,
        verbose=verbose,
        cache_path=valid_cache,
        save_cache=save_cache,
        max_nodes=max_nodes,
    )

    return train_dataset, valid_dataset


def make_spec_datasets(
    cfg,
    spectra_encoder: nn.Module,
    device: torch.device,
    save_cache: bool,
    verbose: bool,
):
    """
    Build train and validation spectra-graph datasets from config.

    Reads the data CSV specified in `cfg.dataset.data_path`, splits it according to
    `cfg.dataset.split_column` / `cfg.dataset.val_tag`, and creates a `GraphSpecDataset`
    for each split. The `spectra_encoder` is passed to the dataset so that fingerprints
    can be pre-computed at dataset construction time rather than on-the-fly.

    Args:
        cfg (DictConfig): OmegaConf config object. Expected keys under `cfg.dataset`:
            `data_path`, `split_column`, `val_tag`, `spec_folder`, `subform_folder`,
            `smile_column`, and optionally `cache_path` and `max_nodes`.
        spectra_encoder (nn.Module): Pre-loaded spectra encoder model used to compute
            fingerprints from MS/MS spectra inside the dataset.
        device (torch.device): Device on which the encoder and tensors will reside.
        save_cache (bool): Whether to write dataset cache files. Set to False on non-main
            ranks in distributed training.
        verbose (bool): Whether to print progress information.

    Returns:
        tuple[GraphSpecDataset, GraphSpecDataset]: A tuple (train_dataset, valid_dataset).
    """
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

    max_nodes = cfg.dataset.get("max_nodes")

    train_dataset = GraphSpecDataset(
        data=train_df,
        cache_path=train_cache,
        save_cache=save_cache,
        spec_folder=cfg.dataset.spec_folder,
        subform_folder=cfg.dataset.subform_folder,
        spectra_encoder=spectra_encoder,
        device=device,
        smile_column=cfg.dataset.smile_column,
        sanitizer=graph_to_smiles_digress,
        verbose=verbose,
        max_nodes=max_nodes,
    )
    valid_elements = train_dataset.valid_elements
    valid_dataset = GraphSpecDataset(
        data=valid_df,
        valid_elements=valid_elements,
        cache_path=valid_cache,
        save_cache=save_cache,
        spec_folder=cfg.dataset.spec_folder,
        subform_folder=cfg.dataset.subform_folder,
        spectra_encoder=spectra_encoder,
        device=device,
        smile_column=cfg.dataset.smile_column,
        sanitizer=graph_to_smiles_digress,
        verbose=verbose,
        max_nodes=max_nodes,
    )

    return train_dataset, valid_dataset
