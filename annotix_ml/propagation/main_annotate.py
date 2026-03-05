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
import json

from annotix_ml.propagation.inference import annotate, annotate_batch, export_results
from load_spectra import load_spectra
from store import json_to_graph

def get_arguments():
    """
    Parse command-line arguments for source databases and output options.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = ArgumentParser(prog="Annotix")
    parser.add_argument("--query_path", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    
    args = get_arguments()

    # LOAD JSON NETWORK
    print("Load the network")
    with open("net.json", "r") as f:
        network = json_to_graph(json.load(f))

    # LOAD UNKNOWN SPECTRA
    query_spectra = load_spectra(Path(args.query_path))

    # ANNOTATE A SINGLE SPECTRUM
    if query_spectra:
        print("\n--- Single spectrum annotation ---")
        result_df = annotate(
            query=query_spectra[0],
            network=network,
            direct_score_threshold=0.7,
            direct_min_matched=6,
            prop_score_threshold=0.5,
            prop_min_matched=4,
            max_depth=2,
            decay_factor=0.7,
            mz_tolerance=0.02,
            top_n=10,
        )
        print(result_df.to_string(index=False))

    # ANNOTATE ALL UNKNOWNS (batch)
    print("\n--- Batch annotation ---")
    batch_results = annotate_batch(
        queries=query_spectra,
        network=network,
        direct_score_threshold=0.7,
        direct_min_matched=6,
        prop_score_threshold=0.5,
        prop_min_matched=4,
        max_depth=2,
        decay_factor=0.7,
        top_n=5,  # top 5 per query
    )

    # EXPORT
    export_results(batch_results, "annotations.csv", format="tsv")

    # Quick summary
    n_total = batch_results["query_id"].nunique()
    n_direct = batch_results[batch_results["source"] == "direct_match"]["query_id"].nunique()
    n_propagated = batch_results[batch_results["source"] == "propagation"]["query_id"].nunique()
    print(f"\nSummary: {n_total} queries, {n_direct} with direct matches, {n_propagated} with propagated annotations")
