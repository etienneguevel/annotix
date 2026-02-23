from annotix_ml.graphtransf.data.dataset import GraphDatasetFromSMILEs
from annotix_ml.graphtransf.data.data_utils import graph_to_smiles_digress


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
