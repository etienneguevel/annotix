import torch
from torch import nn

from annotix_ml.spectraencoder.model.spectra_encoder import SpectraEncoder


def spectra_fingerprint(
    num_peaks: torch.Tensor,
    types: torch.Tensor,
    instruments: torch.Tensor,
    ion_vec: torch.Tensor,
    form_vec: torch.Tensor,
    intens: torch.Tensor,
    spectra_encoder: SpectraEncoder,
    projection: nn.Module,
) -> torch.Tensor:
    """
    Compute the fingerprint of a spectra within a batch using a SpectraEncoder.

    This function mimics the idea of extra feature functions in
    `annotix_ml.graphtransf.math.extra_features`, providing a global feature
    (the fingerprint) for the given spectral data.

    Args:
        num_peaks (torch.Tensor): Number of peaks for each spectrum (bs).
        types (torch.Tensor): Peak types for each spectrum (bs, max_len).
        instruments (torch.Tensor): Instrument indices for each spectrum (bs).
        ion_vec (torch.Tensor): Ion/adduct indices for each peak (bs, max_len).
        form_vec (torch.Tensor): Formula embeddings for each peak (bs, max_len, formula_dim).
        intens (torch.Tensor): Intensity values for each peak (bs, max_len).
        spectra_encoder (SpectraEncoder): Trained SpectraEncoder model instance.

    Returns:
        torch.Tensor: Spectra fingerprints of shape (bs, output_size).
    """
    # Remake the batch dictionary expected by SpectraEncoder
    batch = {
        "num_peaks": num_peaks,
        "types": types,
        "instruments": instruments,
        "ion_vec": ion_vec,
        "form_vec": form_vec,
        "intens": intens,
    }

    # The SpectraEncoder expects the batch dictionary directly
    with torch.no_grad():
        # SpectraEncoder.forward returns (output, aux_outputs)
        enc_output, _ = spectra_encoder(batch)
        fingerprint = projection(enc_output)

    return fingerprint
