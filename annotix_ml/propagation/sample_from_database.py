import bigjson
import json
import numpy as np
from matchms import Spectrum
from matchms.similarity import ModifiedCosine
from load_spectra import load_spectra
from argparse import ArgumentParser

_KEY_MAP = {"PRECURSORMZ": "precursor_mz"}
NR_SIMILAR = 5
THRESHOLD = .7

def get_arguments():
    parser = ArgumentParser(prog="Annotix")
    parser.add_argument("--find_matches", action="store_true")
    return parser.parse_args()

def _parse_peaks(peaks_str: str):
    pairs = [p.split() for p in peaks_str.strip().split(";") if p.strip()]
    mz = np.array([float(p[0]) for p in pairs])
    intensities = np.array([float(p[1]) for p in pairs])
    return mz, intensities

def dict_to_spectrum(entry):
    peaks_str = entry.get("PEAKS_LIST", "")
    if not peaks_str:
        return None
    mz, intensities = _parse_peaks(peaks_str)
    metadata = {_KEY_MAP.get(k, k.lower()): v for k, v in entry.items() if k != "PEAKS_LIST"}
    return Spectrum(mz=mz, intensities=intensities, metadata=metadata)

def find_matches():
    # load 3 queries
    queries = load_spectra("resources/pesticides_test.mgf")

    match_positions = {q.metadata.get('spectrum_id'): [] for q in queries}

    with open("resources/POS_LC.json", "rb") as f:
        data = bigjson.load(f)
        for position, entry in enumerate(data):
            ref = dict_to_spectrum(entry)
            ref_id = ref.metadata.get('spectrum_id')
            for query in queries:
                spec_id = query.metadata.get('spectrum_id')
                if len(match_positions[spec_id]) >= NR_SIMILAR:
                    continue
                sim = ModifiedCosine().pair(reference=ref, query=query)
                if sim["score"] > THRESHOLD:
                    print(f"Query {spec_id} matched {ref_id} (score: {sim['score']}, matches: {sim['matches']}). Position: {position}")
                    match_positions[spec_id].append(position)

            if {len(val) for val in match_positions.values()} == {NR_SIMILAR}:
                print("Found all similar items. Stopping now...")
                break

    print(match_positions)

    with open("match_positions.json", "w") as f:
        json.dump(match_positions, f)

def extract_sample():

    # Load the matched positions
    with open("match_positions.json", "r") as f:
        match_positions = json.load(f)

    positions = []
    for pos in match_positions.values():
        positions.extend(pos)

    # Select other random positions and add it to the matched ones
    indexes = np.unique(np.sort([int(item) for item in np.random.choice(a=range(2000000), size=200, replace=False)] + positions)).tolist()

    with open("resources/POS_LC.json", "rb") as f:
        data = bigjson.load(f)
        sample = []
        for pos in indexes:
            print(f"Process item {pos}")
            try:
                item = data[pos].to_python()
                if not item.get("SPECTRUMID"):
                    print("  Add spectrum ID")
                    item["SPECTRUMID"] = f"spec_{pos}"
                sample.append(item)
            except Exception as err:
                print("   Problem:", err)

    print(f"{len(sample)} item processed")

    with open("resources/POS_LC_sample_2.json", "w") as f:
        json.dump(sample, f)

    print("Done")

if __name__ == "__main__":
    
    args = get_arguments()

    if args.find_matches:
        find_matches()

    extract_sample()
