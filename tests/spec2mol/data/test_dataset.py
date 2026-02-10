from annotix_ml.spec2mol.data.dataset import GraphSpecDataset
from annotix_ml.spec2mol.data.datacollator import graph_spec_collate_fn
from torch.utils.data import DataLoader
from annotix_ml import BASE_DIR


def test_graph_spec_dataset():
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

    # Test length
    assert len(dataset) > 0

    # Test __getitem__
    item = dataset[0]
    expected_item_keys = [
        "nodes",
        "edges",
        "smiles",
        "spec_name",
        "peak_type",
        "form_vec",
        "ion_vec",
        "frag_intens",
        "instrument",
    ]
    for key in expected_item_keys:
        assert key in item, f"Missing key in item: {key}"

    # Test Collate
    loader = DataLoader(dataset, batch_size=2, collate_fn=graph_spec_collate_fn)
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

    assert nodes.shape[0] == 2
    assert types.shape[0] == 2
    assert len(smiles) == 2
