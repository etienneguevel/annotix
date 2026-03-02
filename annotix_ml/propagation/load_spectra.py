from pathlib import Path
import json

import numpy as np
from matchms import Spectrum
from matchms.importing import load_from_mgf
from matchms.filtering import default_filters, normalize_intensities, select_by_mz, select_by_relative_intensity, add_precursor_mz


# matchms requires `precursor_mz` (snake_case); JSON uses `PRECURSORMZ`
_KEY_MAP = {"PRECURSORMZ": "precursor_mz"}

def _parse_peaks(peaks_str: str):
    """Parse 'mz intensity;mz intensity;...' into two numpy arrays."""
    pairs = [p.split() for p in peaks_str.strip().split(";") if p.strip()]
    mz = np.array([float(p[0]) for p in pairs])
    intensities = np.array([float(p[1]) for p in pairs])
    return mz, intensities

def load_from_json(path: Path):
    """Yield matchms.Spectrum objects from the Annotix JSON format."""
    with open(path) as f:
        entries = json.load(f)
    for entry in entries:
        peaks_str = entry.get("PEAKS_LIST", "")
        if not peaks_str:
            continue
        mz, intensities = _parse_peaks(peaks_str)
        # normalize keys: apply explicit map, then lowercase everything else
        metadata = {_KEY_MAP.get(k, k.lower()): v for k, v in entry.items() if k != "peaks_json"}
        yield Spectrum(mz=mz, intensities=intensities, metadata=metadata)

def clean_spectrum(spec: Spectrum) -> Spectrum:
    """Apply standard filters to a spectrum."""
    spec = default_filters(spec)
    spec = add_precursor_mz(spec)
    spec = normalize_intensities(spec)
    spec = select_by_mz(spec, mz_from=10.0, mz_to=2000.0)
    spec = select_by_relative_intensity(spec, intensity_from=0.01)  # remove <1% peaks
    return spec

# Load and clean reference spectra
def load_spectra(ref_path):
    print("Loading spectra...")
    if ref_path.suffix == ".mgf":
        raw = list(load_from_mgf(ref_path))
    elif ref_path.suffix == ".json":
        raw = list(load_from_json(ref_path))
    else:
        raise ValueError(f"Unsupported format: {ref_path.suffix} (use .mgf or .json)")
    spec = [clean_spectrum(s) for s in raw]
    spec = [s for s in spec if s is not None and s.peaks.mz.size >= 4]
    print(f"Loaded {len(spec)} spectra")
    return spec
