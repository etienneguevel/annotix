import torch
from torch.utils.data import DataLoader

from annotix_ml import BASE_DIR
from annotix_ml.spec2mol.data.dataset import GraphSpecDataset
from annotix_ml.spec2mol.data.datacollator import graph_spec_collate_fn
from annotix_ml.graphtransf.data.atoms_data import TYPE_EDGES


def test_collate_graph_spec():
    """Test graph_spec_collate_fn produces correctly shaped and padded batches."""
    data_path = BASE_DIR / "data/BanyulsInference/labels.tsv"
    spec_folder = BASE_DIR / "data/BanyulsInference/spec_files"
    subform_folder = "data/BanyulsInference/subformulae/default_subformulae"

    dataset = GraphSpecDataset(
        data=data_path,
        spec_folder=spec_folder,
        subform_folder=subform_folder,
        smile_column="smiles",
        spec_column="spec",
        formula_column="formula",
        instrument_column="instrument",
    )

    bs = 4
    loader = DataLoader(dataset, batch_size=bs, collate_fn=graph_spec_collate_fn)
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
    ) = next(iter(loader))

    # Batch dimension
    assert nodes.shape[0] == bs
    assert edges.shape[0] == bs
    assert mask.shape[0] == bs
    assert num_peaks.shape[0] == bs
    assert types.shape[0] == bs
    assert len(smiles) == bs

    # Tensor ranks
    assert len(nodes.shape) == 3  # (bs, n, natoms)
    assert len(edges.shape) == 4  # (bs, n, n, nbonds)
    assert len(mask.shape) == 2  # (bs, n)

    # Spatial dimensions are consistent
    n = nodes.shape[1]
    assert edges.shape[1] == n
    assert edges.shape[2] == n
    assert mask.shape[1] == n

    # Edge bond dimension matches TYPE_EDGES
    assert edges.shape[-1] == len(TYPE_EDGES)

    # Mask is boolean
    assert mask.dtype == torch.bool

    # Padded nodes have NoBond (index 0) set to 1 in edges
    for b in range(bs):
        n_real = int(mask[b].sum())
        if n_real < n:
            # Padded rows should have NoBond=1 for all columns
            assert (edges[b, n_real:, :, 0] == 1).all(), (
                "Padded rows should have NoBond=1"
            )
            # Padded columns should have NoBond=1 for all rows
            assert (edges[b, :, n_real:, 0] == 1).all(), (
                "Padded columns should have NoBond=1"
            )

    # Nodes in padded positions should be zero
    for b in range(bs):
        n_real = int(mask[b].sum())
        if n_real < n:
            assert (nodes[b, n_real:] == 0).all(), (
                "Padded node positions should be zero"
            )
