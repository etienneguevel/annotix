from pathlib import Path
import json
import multiprocessing as mp
import os
import sys

import numpy as np
from matchms import Spectrum
from matchms.importing import load_from_mgf
from matchms.filtering import default_filters, normalize_intensities, select_by_mz, select_by_relative_intensity, add_precursor_mz


_KEY_MAP = {"PRECURSORMZ": "precursor_mz"}

def _parse_peaks(peaks_str: str):
    pairs = [p.split() for p in peaks_str.strip().split(";") if p.strip()]
    mz = np.array([float(p[0]) for p in pairs])
    intensities = np.array([float(p[1]) for p in pairs])
    return mz, intensities

def load_from_json(path: Path):
    with open(path) as f:
        entries = json.load(f)
    peaks_key = "PEAKS_LIST"
    for entry in entries:
        peaks_str = entry.get(peaks_key, "")
        if not peaks_str:
            continue
        mz, intensities = _parse_peaks(peaks_str)
        metadata = {_KEY_MAP.get(k, k.lower()): v for k, v in entry.items() if k != peaks_key}
        yield Spectrum(mz=mz, intensities=intensities, metadata=metadata)

def clean_spectrum(spec: Spectrum) -> Spectrum:
    # Module-level so it's picklable by multiprocessing
    spec = default_filters(spec)
    spec = add_precursor_mz(spec)
    spec = normalize_intensities(spec)
    spec = select_by_mz(spec, mz_from=10.0, mz_to=2000.0)
    spec = select_by_relative_intensity(spec, intensity_from=0.01)
    return spec

def load_spectra(ref_path, n_workers: int = None):
    print("Loading spectra...")
    if ref_path.suffix == ".mgf":
        raw = list(load_from_mgf(ref_path))
    elif ref_path.suffix == ".json":
        raw = list(load_from_json(ref_path))
    else:
        raise ValueError(f"Unsupported format: {ref_path.suffix} (use .mgf or .json)")

    n_workers = n_workers or os.cpu_count()
    chunksize = max(1, len(raw) // (n_workers * 4))
    ctx = mp.get_context("fork") if sys.platform != "win32" else mp.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        cleaned = pool.map(clean_spectrum, raw, chunksize=chunksize)

    del raw

    spec = [s for s in cleaned if s is not None and s.peaks.mz.size >= 4]
    print(f"Loaded {len(spec)} spectra")
    return spec
