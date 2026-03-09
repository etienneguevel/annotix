import bigjson
import json
from random import randint
import numpy as np
from matchms import Spectrum
from matchms.similarity import ModifiedCosine
from load_spectra import load_spectra

_KEY_MAP = {"PRECURSORMZ": "precursor_mz"}
NR_SIMILAR = 5
THRESHOLD = .7

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


def run():
    # load 3 queries
    queries = load_spectra("resources/pesticides_test.mgf")

    match_positions = {}

    for query in queries:
        spec_id = query.metadata.get('spectrum_id')
        print(f"\n\033[95m*** Processing {spec_id} ***\033[0m")

        with open("resources/POS_LC.json", "rb") as f:
            data = bigjson.load(f)
            positions = []
            for position, entry in enumerate(data):
                ref = dict_to_spectrum(entry)
                sim = ModifiedCosine().pair(reference=ref, query=query)
                score = sim["score"]
                matches = sim["matches"]
                if score > THRESHOLD:
                    print(f"Match found: {ref.metadata.get('spectrum_id')}. Position: {position}. Score: {score}. Peak matches: {matches}")
                    positions.append(position)
                if len(positions) >= NR_SIMILAR:
                    break
        match_positions[spec_id] = positions


    print(match_positions)


    # parse le big json
    # calcul des similarité
    # on récupère top_5 similaires
    # on ajoute des autres molécules random
    # on stocke



    # other = [21485, 22006, 22009, 129179, 129349, 132470, 140261, 142300]
    # indexes = set([randint(0, i) for i in range(2000000)][:200] + other)

    # with open("resources/POS_LC.json", "rb") as f:
    #     data = bigjson.load(f)
    #     sample = []
    #     for pos in indexes:
    #         print(f"Process item {pos}")
    #         try:
    #             item = data[pos].to_python()
    #             if not item.get("SPECTRUMID"):
    #                 print("  Add spectrum ID")
    #                 item["SPECTRUMID"] = f"spec_{pos}"
    #             sample.append(item)
    #         except Exception as err:
    #             print("   Problem:", err)

    # print(f"{len(sample)} item processed")

    # with open("resources/POS_LC_sample_2.json", "w") as f:
    #     json.dump(sample, f)

if __name__ == "__main__":
    run()

