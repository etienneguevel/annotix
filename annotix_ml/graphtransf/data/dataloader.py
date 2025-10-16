from torch.utils.data import DataLoader

from annotix_ml.graphtransf.data.datacollator import collateGraph

# TODO : think about the utility of this function ? To test data_collator ?
def get_loader(dataset, batch_size: int, shuffle: bool = True):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collateGraph,
    )

    return loader
