"""
Final annotation script for unknown molecules

- pass a batch of MS/MS spectra
- loop over the most similar molecules found in the molécular network
- annotate each molecule
"""
from argparse import ArgumentParser
from pathlib import Path
import json

from annotix_ml.propagation.inference import annotate, select_annotation
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
    parser.add_argument("--network_path", type=str, default="net.json")
    return parser.parse_args()

if __name__ == "__main__":
    
    args = get_arguments()

    # LOAD JSON NETWORK
    print("Load the network")
    with open(args.network_path, "r") as f:
        network = json_to_graph(json.load(f))

    # LOAD UNKNOWN SPECTRA
    query_spectra = load_spectra(Path(args.query_path))

    # Infer each new molecule
    for query in query_spectra:
        print(f"\n*** Annotating {query.get('spectrum_id')} ***")
        result_df = annotate(
            query=query,
            network=network,
            direct_score_threshold=0.7,
            direct_min_matched=6,
            prop_score_threshold=0.5,
            prop_min_matched=4,
            max_depth=2,
            decay_factor=0.7,
            mz_tolerance=0.02,
            top_n=10
        )

        decision = select_annotation(result_df)
        print(decision)
