"""
Example Usage
=============
Shows how to use the spectral annotation pipeline with:
  (a) GNPS .mgf library files
  (b) Your unknown spectra

Adapt the file paths to your data.
"""
from argparse import ArgumentParser
from pathlib import Path
from time import time

from pipeline import build_molecular_network
from load_spectra import load_spectra
from plot import draw


def get_arguments():
    """
    Parse command-line arguments for source databases and output options.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = ArgumentParser(prog="Annotix")
    parser.add_argument("--ref_path", type=str)
    parser.add_argument("--n_workers", type=int, default=4)
    return parser.parse_args()

if __name__ == "__main__":

    # The input files should be of MGF format, from GNPS
    # Can be downloaded from: https://gnps-external.ucsd.edu/gnpslibrary, e.g., GNPS-LIBRARY.mgf or ALL_GNPS.mgf

    args = get_arguments()

    start = time()

    # STEP 1: LOAD REFERENCE SPECTRA (your molecular network database)
    reference_spectra = load_spectra(Path(args.ref_path), n_workers=args.n_workers)

    current = time()
    duration_seconds = current - start
    print(f"Load spectra done in {duration_seconds:.2f} seconds ({duration_seconds / 3600:.2f} hours)")

    # STEP 2: BUILD THE MOLECULAR NETWORK
    print("Building molecular network...")
    network = build_molecular_network(
        spectra=reference_spectra,
        cosine_threshold=0.7,
        min_matched_peaks=6,
        top_k=10,
        mz_tolerance=0.02,
        path="./net.json",
        n_workers=args.n_workers
    )

    end = time()
    duration_seconds = end - current
    print(f"Build network done in {duration_seconds:.2f} seconds ({duration_seconds / 3600:.2f} hours)")

    draw(network)
