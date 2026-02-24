import pandas as pd
from torch.utils.data import DataLoader

from annotix_ml import BASE_DIR
from annotix_ml.data.datacollator import collateGraph
from annotix_ml.data.dataset import GraphDatasetFromSMILEs


def test_collate_MSG():
    # Open the MSG dataset
    df = pd.read_csv(BASE_DIR / "data" / "MassSpecGym.csv")

    # Select a part of the dataset
    df = df[df.fold == "train"].sample(100)

    # Create the dataset & dataloader
    bs = 16
    dataset = GraphDatasetFromSMILEs(df)

    loader = DataLoader(dataset, batch_size=bs, collate_fn=collateGraph)

    N, E, node_mask = next(iter(loader))

    assert N.shape[0] == bs
    assert E.shape[0] == bs
    assert node_mask.shape[0] == bs
    assert len(N.shape) == 3
    assert len(E.shape) == 4
