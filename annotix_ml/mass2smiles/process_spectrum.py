
import matchms.filtering as filt

def spectrum_processing(s):
    """This is how one would typically design a desired pre- and post-
    processing pipeline."""
    s = filt.default_filters(s)
    s = filt.add_parent_mass(s)
    s = filt.normalize_intensities(s)
    s = filt.select_by_intensity(s, intensity_from=0.01)
    s = filt.reduce_to_number_of_peaks(s, n_required=5, n_max=250)
    s = filt.select_by_mz(s, mz_from=15, mz_to=2000)
    s = filt.add_losses(s, loss_mz_from=15.0, loss_mz_to=350.0)
    s = filt.require_minimum_number_of_peaks(s, n_required=5)
    return s

def metadata_processing(spectrum):
    spectrum = filt.default_filters(spectrum)
    spectrum = filt.repair_inchi_inchikey_smiles(spectrum)
    spectrum = filt.derive_inchi_from_smiles(spectrum)
    spectrum = filt.derive_smiles_from_inchi(spectrum)
    spectrum = filt.derive_inchikey_from_inchi(spectrum)
    spectrum = filt.harmonize_undefined_smiles(spectrum)
    spectrum = filt.harmonize_undefined_inchi(spectrum)
    spectrum = filt.harmonize_undefined_inchikey(spectrum)
    return spectrum
