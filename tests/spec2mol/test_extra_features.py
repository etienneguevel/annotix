import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from annotix_ml import BASE_DIR
from annotix_ml.spec2mol.data.dataset import GraphSpecDataset
from annotix_ml.spec2mol.data.datacollator import graph_spec_collate_fn
from annotix_ml.spec2mol.extra_features import spectra_fingerprint
from annotix_ml.spectraencoder.model.spectra_encoder import SpectraEncoder


def test_spectra_fingerprint():
    """Test that spectra_fingerprint produces correctly shaped output."""
    cfg = OmegaConf.load(BASE_DIR / "configs" / "default_config.yaml")

    # Load a small batch from the real dataset
    dataset = GraphSpecDataset(
        data=BASE_DIR / cfg.dataset.labels_file,
        spec_folder=BASE_DIR / cfg.dataset.spec_folder,
        subform_folder=cfg.dataset.subform_folder,
        smile_column="smiles",
        spec_column="spec",
        formula_column="formula",
        instrument_column="instrument",
    )

    bs = 2
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

    # Build the encoder and projection
    output_size = cfg.spectra_encoder.output_size
    morgan_nbits = cfg.dataset.morgan_nbits

    encoder = SpectraEncoder(
        form_embedder=cfg.spectra_encoder.form_embedder,
        output_size=output_size,
        hidden_size=cfg.spectra_encoder.hidden_size,
        spectra_dropout=cfg.spectra_encoder.spectra_dropout,
        top_layers=cfg.spectra_encoder.top_layers,
        magma_modulo=cfg.spectra_encoder.magma_modulo,
        peak_attn_layers=cfg.spectra_encoder.peak_attn_layers,
        set_pooling=cfg.spectra_encoder.set_pooling,
        pairwise_featurization=cfg.spectra_encoder.pairwise_featurization,
        num_heads=cfg.spectra_encoder.num_heads,
        embed_instrument=cfg.spectra_encoder.embed_instrument,
        inten_transform=cfg.spectra_encoder.inten_transform,
        no_diffs=cfg.spectra_encoder.no_diffs,
    )
    projection = torch.nn.Linear(output_size, morgan_nbits)

    # Compute the fingerprint
    fp = spectra_fingerprint(
        num_peaks=num_peaks,
        types=types,
        instruments=instruments,
        ion_vec=ion_vec,
        form_vec=form_vec,
        intens=intens,
        spectra_encoder=encoder,
        projection=projection,
    )

    # Shape: (bs, morgan_nbits)
    assert fp.shape == (bs, morgan_nbits), (
        f"Expected ({bs}, {morgan_nbits}), got {fp.shape}"
    )

    # Output should not require grad (computed under torch.no_grad)
    assert not fp.requires_grad


def test_spectra_fingerprint_deterministic():
    """Test that spectra_fingerprint is deterministic (same input -> same output)."""
    cfg = OmegaConf.load(BASE_DIR / "configs" / "default_config.yaml")

    dataset = GraphSpecDataset(
        data=BASE_DIR / cfg.dataset.labels_file,
        spec_folder=BASE_DIR / cfg.dataset.spec_folder,
        subform_folder=cfg.dataset.subform_folder,
        smile_column="smiles",
        spec_column="spec",
        formula_column="formula",
        instrument_column="instrument",
    )

    loader = DataLoader(dataset, batch_size=2, collate_fn=graph_spec_collate_fn)
    batch = next(iter(loader))
    _, _, _, num_peaks, types, instruments, ion_vec, form_vec, intens, _ = batch

    encoder = SpectraEncoder(
        form_embedder=cfg.spectra_encoder.form_embedder,
        output_size=cfg.spectra_encoder.output_size,
        hidden_size=cfg.spectra_encoder.hidden_size,
        spectra_dropout=0.0,  # No dropout for determinism
        top_layers=cfg.spectra_encoder.top_layers,
        magma_modulo=cfg.spectra_encoder.magma_modulo,
        peak_attn_layers=cfg.spectra_encoder.peak_attn_layers,
        set_pooling=cfg.spectra_encoder.set_pooling,
        pairwise_featurization=cfg.spectra_encoder.pairwise_featurization,
        num_heads=cfg.spectra_encoder.num_heads,
        embed_instrument=cfg.spectra_encoder.embed_instrument,
        inten_transform=cfg.spectra_encoder.inten_transform,
        no_diffs=cfg.spectra_encoder.no_diffs,
    )
    encoder.eval()
    projection = torch.nn.Linear(
        cfg.spectra_encoder.output_size, cfg.dataset.morgan_nbits
    )

    kwargs = dict(
        num_peaks=num_peaks,
        types=types,
        instruments=instruments,
        ion_vec=ion_vec,
        form_vec=form_vec,
        intens=intens,
        spectra_encoder=encoder,
        projection=projection,
    )

    fp1 = spectra_fingerprint(**kwargs)
    fp2 = spectra_fingerprint(**kwargs)

    assert torch.equal(fp1, fp2), "spectra_fingerprint should be deterministic"
