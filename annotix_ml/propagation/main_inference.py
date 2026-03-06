"""
Final annotation script for unknown molecules

- pass a batch of MS/MS spectra
- loop over the most similar molecules found in the molécular network
- annotate each molecule
"""
from argparse import ArgumentParser
from pathlib import Path
import json
import matplotlib.pyplot as plt

from annotix_ml.propagation.inference import find_candidates, infer_structure
from load_spectra import load_spectra
from store import json_to_graph
from plot import plot_spectrum

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
        print(f"\n\033[95m*** Annotating {query.get('spectrum_id')} ***\033[0m")
        result_df = find_candidates(
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

        # Check for direct matches in the candidates molecules
        for idx, row in result_df.iterrows():
            if row["score"] == 1 and row["source"] == "direct_match" and not row["mass_shift_da"] and row["compound_name"] != "Unknown":
                print(f"[Exact match] {row.get('compound_name', '?')} → {row["smiles"]}")

                plot_spectrum(query, row)
                plt.show()

        print(result_df)

        with open("resources/delta_mass_table.json", "r") as f:
            delta_mass_table = json.load(f)

        all_candidates = infer_structure(query=query, result_df=result_df, network=network, delta_mass_table=delta_mass_table, delta_mass_tolerance=0.05)
        if all_candidates:
            print(all_candidates)
