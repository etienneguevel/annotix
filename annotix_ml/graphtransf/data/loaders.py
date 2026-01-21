from annotix_ml.graphtransf.data.dataset import GraphDatasetFromSMILEs
from annotix_ml.graphtransf.data.data_utils import graph_to_smiles_digress


def make_datasets(
    data_path: str,
    smile_column: str = "smiles",
    split_column: str = "fold",
    val_tag: str | None = "test",
):
    # Make the datasets
    train_dataset = GraphDatasetFromSMILEs(
        data=data_path,
        smile_column=smile_column,
        split="train",
        split_column=split_column,
        sanitizer=graph_to_smiles_digress,
    )
    valid_dataset = GraphDatasetFromSMILEs(
        data=data_path,
        smile_column=smile_column,
        split=val_tag,
        split_column=split_column,
        sanitizer=graph_to_smiles_digress,
    )

    return train_dataset, valid_dataset
