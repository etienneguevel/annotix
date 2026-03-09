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
from plot import plot_spectrum, plot_proposals

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

    with open("resources/delta_mass_table.json", "r") as f:
        delta_mass_table = json.load(f)

    # LOAD UNKNOWN SPECTRA
    query_spectra = load_spectra(Path(args.query_path))

    # Infer each new molecule
    for query in query_spectra:
        print(f"\n\033[95m*** Annotating {query.get('spectrum_id')} ***\033[0m")
        true_smiles = query.metadata.get("smiles")
        # print(f"True SMILES: {true_smiles}")

        result_df = find_candidates(
            query=query,
            network=network,
            direct_score_threshold=.7,
            direct_min_matched=6,
            prop_score_threshold=.5,
            prop_min_matched=4,
            max_depth=2,
            decay_factor=.7,
            mz_tolerance=.02,
            top_n=10
        )

        # Correct the confidence level for exact matches / should be in the function find_candidates
        result_df.loc[(result_df["score"] == 1) & (result_df["source"] == "direct_match") & (result_df["mass_shift_da"] == 0) & (result_df["compound_name"] != "Unknown"), "confidence_level"] = "Exact match"

        # Remove Unknown molecules
        result_df = result_df[result_df["compound_name"] != "Unknown"]

        # Check for exact matches in the candidates molecules
        if "Exact match" in result_df["confidence_level"].values:
            exact_matches = result_df[result_df["confidence_level"] == "Exact match"]
            print(f"  Found {len(exact_matches)} exact matche(s)")
            for _, row in exact_matches.iterrows():
                print(f"  {row.get('compound_name', '?')} → {row["smiles"]}")
                plot_spectrum(query, row)
                plt.show()

        else:
            all_candidates = infer_structure(query=query, result_df=result_df, network=network, delta_mass_table=delta_mass_table, delta_mass_tolerance=0.05)
            if all_candidates:
                ref_row = result_df.iloc[0] if not result_df.empty else None
                plot_proposals(query, all_candidates, reference_row=ref_row, true_smiles=true_smiles)
