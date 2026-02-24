import pandas as pd

from annotix_ml.data.dataset import GraphDatasetFromSMILEs
from annotix_ml.data.spec_dataset import GraphSpecDataset
from annotix_ml.data.data_utils import graph_to_smiles_digress


def make_datasets(
    data_path: str,
    smile_column: str = "smiles",
    split_column: str = "fold",
    val_tag: str | None = "test",
    verbose: bool = True,
    cache_path: str | None = None,
    save_cache: bool = True,
):
    # Derive per-split cache paths from the base cache_path prefix
    train_cache = f"{cache_path}_train.pt" if cache_path is not None else None
    valid_cache = f"{cache_path}_valid.pt" if cache_path is not None else None

    # Make the datasets
    train_dataset = GraphDatasetFromSMILEs(
        data=data_path,
        smile_column=smile_column,
        split="train",
        split_column=split_column,
        sanitizer=graph_to_smiles_digress,
        verbose=verbose,
        cache_path=train_cache,
        save_cache=save_cache,
    )
    valid_dataset = GraphDatasetFromSMILEs(
        data=data_path,
        smile_column=smile_column,
        split=val_tag,
        split_column=split_column,
        sanitizer=graph_to_smiles_digress,
        verbose=verbose,
        cache_path=valid_cache,
        save_cache=save_cache,
    )

    return train_dataset, valid_dataset


def make_spec_datasets(cfg, save_cache: bool):
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
        save_cache=save_cache,
        **common_kwargs,
    )
    valid_elements = train_dataset.valid_elements
    valid_dataset = GraphSpecDataset(
        data=valid_df,
        valid_elements=valid_elements,
        cache_path=valid_cache,
        save_cache=save_cache,
        **common_kwargs,
    )

    return train_dataset, valid_dataset
