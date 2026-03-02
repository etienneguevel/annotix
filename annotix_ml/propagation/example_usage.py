"""
Example Usage
=============
Shows how to use the spectral annotation pipeline with:
  (a) GNPS .mgf library files
  (b) Your unknown spectra

Adapt the file paths to your data.
"""

from matchms import Spectrum
from matchms.importing import load_from_mgf
from matchms.filtering import (
    default_filters,
    normalize_intensities,
    select_by_mz,
    select_by_relative_intensity,
    add_precursor_mz,
)
from pipeline import build_molecular_network, annotate, annotate_batch, export_results, write_adjacency


# =============================================================================
# STEP 0: SPECTRUM CLEANING FUNCTION
# =============================================================================

def clean_spectrum(spec: Spectrum) -> Spectrum:
    """Apply standard filters to a spectrum."""
    spec = default_filters(spec)
    spec = add_precursor_mz(spec)
    spec = normalize_intensities(spec)
    spec = select_by_mz(spec, mz_from=10.0, mz_to=2000.0)
    spec = select_by_relative_intensity(spec, intensity_from=0.01)  # remove <1% peaks
    return spec


# =============================================================================
# STEP 1: LOAD REFERENCE SPECTRA (your molecular network database)
# =============================================================================
# Download a GNPS library .mgf from:
#   https://gnps-external.ucsd.edu/gnpslibrary
# e.g., GNPS-LIBRARY.mgf or ALL_GNPS.mgf

# REFERENCE_MGF = "/Users/baptistegregorutti/Documents/Projets/En cours/Annotix/annotix-ml/main/test_data/mzmine_export_iimn_gnps.mgf"
REFERENCE_MGF = "../../resources/pesticides.mgf"
QUERY_MGF = "../../resources/pesticides.mgf"

# Load and clean reference spectra
print("Loading reference spectra...")
reference_spectra = list(load_from_mgf(REFERENCE_MGF)) #OR USE Spectrum CLASS FOR JSON DATA
reference_spectra = [clean_spectrum(s) for s in reference_spectra]
reference_spectra = [s for s in reference_spectra if s is not None and s.peaks.mz.size >= 6]  # need enough fragments
print(f"Loaded {len(reference_spectra)} reference spectra")


# =============================================================================
# STEP 2: BUILD THE MOLECULAR NETWORK
# =============================================================================

print("Building molecular network...")
network = build_molecular_network(
    spectra=reference_spectra,
    cosine_threshold=0.7,
    min_matched_peaks=6,
    top_k=10,
    mz_tolerance=0.02,
)

# # =============================================================================
# # STEP 3: LOAD UNKNOWN SPECTRA
# # =============================================================================

# print("Loading query spectra...")
# query_spectra = list(load_from_mgf(QUERY_MGF))
# query_spectra = [clean_spectrum(s) for s in query_spectra]
# query_spectra = [s for s in query_spectra if s is not None and s.peaks.mz.size >= 4]
# print(f"Loaded {len(query_spectra)} query spectra")


# # =============================================================================
# # STEP 4a: ANNOTATE A SINGLE SPECTRUM
# # =============================================================================

# if query_spectra:
#     print("\n--- Single spectrum annotation ---")
#     result_df = annotate(
#         query=query_spectra[0],
#         network=network,
#         direct_score_threshold=0.7,
#         direct_min_matched=6,
#         prop_score_threshold=0.5,
#         prop_min_matched=4,
#         max_depth=2,
#         decay_factor=0.7,
#         mz_tolerance=0.02,
#         top_n=10,
#     )
#     print(result_df.to_string(index=False))
#     print(result_df.loc[0])


# # =============================================================================
# # STEP 4b: ANNOTATE ALL UNKNOWNS (batch)
# # =============================================================================

# print("\n--- Batch annotation ---")
# batch_results = annotate_batch(
#     queries=query_spectra,
#     network=network,
#     direct_score_threshold=0.7,
#     direct_min_matched=6,
#     prop_score_threshold=0.5,
#     prop_min_matched=4,
#     max_depth=2,
#     decay_factor=0.7,
#     top_n=5,  # top 5 per query
# )


# # =============================================================================
# # STEP 5: EXPORT
# # =============================================================================

# export_results(batch_results, "annotations.csv", format="tsv")

# # Quick summary
# n_total = batch_results["query_id"].nunique()
# n_direct = batch_results[batch_results["source"] == "direct_match"]["query_id"].nunique()
# n_propagated = batch_results[batch_results["source"] == "propagation"]["query_id"].nunique()
# print(f"\nSummary: {n_total} queries, {n_direct} with direct matches, "
#       f"{n_propagated} with propagated annotations")
