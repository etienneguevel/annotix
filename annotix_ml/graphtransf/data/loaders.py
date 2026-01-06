from annotix_ml.graphtransf.data.dataset import GraphDatasetFromSMILEs


def make_datasets(
    data_path: str,
    smile_column: str = "smiles",
    split_column: str = "fold",
    val_tag: str = "val",
):
    # Make the datasets
    train_dataset = GraphDatasetFromSMILEs(
        data=data_path,
        smile_column=smile_column,
        split="train",
        split_column=split_column,
    )
    valid_dataset = GraphDatasetFromSMILEs(
        data=data_path,
        smile_column=smile_column,
        split=val_tag,
        split_column=split_column,
    )

    return train_dataset, valid_dataset
