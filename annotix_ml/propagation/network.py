"""
Molecular network building functions
====================================

A molecular network: a NetworkX graph where nodes carry MS/MS spectra
and (some) annotations from a reference library (e.g., GNPS)
"""

import heapq
import multiprocessing as mp
from multiprocessing import shared_memory
import os
import sys
from collections import OrderedDict

import networkx as nx
import numpy as np
from dataclasses import dataclass
from typing import Optional
from matchms import Spectrum
from matchms.similarity import ModifiedCosine
from tqdm import tqdm
import json

from scipy import sparse

from .store import json_to_graph, _serialize_value


# =============================================================================
# PARALLEL PAIRWISE SIMILARITY HELPERS (shared-memory version)
# =============================================================================

def _mp_context():
    """Return a fork-based multiprocessing context when possible."""
    if sys.platform != "win32":
        return mp.get_context("fork")
    return mp.get_context("spawn")


def _pack_into_shared_memory(
    spectra: list[Spectrum],
) -> tuple[shared_memory.SharedMemory, shared_memory.SharedMemory, shared_memory.SharedMemory, np.ndarray]:
    all_mz = []
    all_int = []
    offsets = [0]
    for s in spectra:
        mz = np.asarray(s.mz, dtype=np.float64)
        ints = np.asarray(s.intensities, dtype=np.float64)
        all_mz.append(mz)
        all_int.append(ints)
        offsets.append(offsets[-1] + len(mz))
    flat_mz = np.concatenate(all_mz) if all_mz else np.array([], dtype=np.float64)
    flat_int = np.concatenate(all_int) if all_int else np.array([], dtype=np.float64)
    offsets = np.array(offsets, dtype=np.int64)

    shm_mz = shared_memory.SharedMemory(create=True, size=max(8, flat_mz.nbytes))
    shm_int = shared_memory.SharedMemory(create=True, size=max(8, flat_int.nbytes))
    shm_off = shared_memory.SharedMemory(create=True, size=max(8, offsets.nbytes))

    np.ndarray(flat_mz.shape, dtype=np.float64, buffer=shm_mz.buf)[:] = flat_mz
    np.ndarray(flat_int.shape, dtype=np.float64, buffer=shm_int.buf)[:] = flat_int
    np.ndarray(offsets.shape, dtype=np.int64, buffer=shm_off.buf)[:] = offsets

    return shm_mz, shm_int, shm_off, offsets


# --- Per-worker globals (attached via initializer, not pickled) ---
_shm_mz_view: np.ndarray | None = None
_shm_int_view: np.ndarray | None = None
_shm_off_view: np.ndarray | None = None
_shm_handles: list = []
_worker_n: int = 0
_worker_pmzs: np.ndarray | None = None
_worker_cache: OrderedDict | None = None
_worker_fp: sparse.csr_matrix | None = None
_worker_fp_thresh: float = 0.0
_CACHE_MAX = 256


def _init_worker_shm(
    shm_mz_name: str, shm_int_name: str, shm_off_name: str,
    total_peaks: int, n: int, precursor_mzs: list[float],
    fp_shm_names: tuple[str, str, str] | None = None,
    fp_shapes: tuple[int, int, int, int] | None = None,
    fingerprint_threshold: float = 0.0,
):
    global _shm_mz_view, _shm_int_view, _shm_off_view, _shm_handles
    global _worker_n, _worker_pmzs, _worker_cache
    global _worker_fp, _worker_fp_thresh

    shm_mz = shared_memory.SharedMemory(name=shm_mz_name, create=False)
    shm_int = shared_memory.SharedMemory(name=shm_int_name, create=False)
    shm_off = shared_memory.SharedMemory(name=shm_off_name, create=False)
    _shm_handles = [shm_mz, shm_int, shm_off]

    _shm_mz_view = np.ndarray((total_peaks,), dtype=np.float64, buffer=shm_mz.buf)
    _shm_int_view = np.ndarray((total_peaks,), dtype=np.float64, buffer=shm_int.buf)
    _shm_off_view = np.ndarray((n + 1,), dtype=np.int64, buffer=shm_off.buf)
    _worker_n = n
    _worker_pmzs = np.array(precursor_mzs, dtype=np.float64)
    _worker_cache = OrderedDict()

    _worker_fp = None
    _worker_fp_thresh = fingerprint_threshold
    if fp_shm_names is not None and fp_shapes is not None:
        nnz, n_rows, n_cols = fp_shapes[0], fp_shapes[1], fp_shapes[2]
        shm_d = shared_memory.SharedMemory(name=fp_shm_names[0], create=False)
        shm_i = shared_memory.SharedMemory(name=fp_shm_names[1], create=False)
        shm_p = shared_memory.SharedMemory(name=fp_shm_names[2], create=False)
        _shm_handles.extend([shm_d, shm_i, shm_p])
        data = np.ndarray((nnz,), dtype=np.float64, buffer=shm_d.buf)
        indices = np.ndarray((nnz,), dtype=np.int32, buffer=shm_i.buf)
        indptr = np.ndarray((n_rows + 1,), dtype=np.int64, buffer=shm_p.buf)
        _worker_fp = sparse.csr_matrix((data, indices, indptr), shape=(n_rows, n_cols))


def _get_spectrum(i: int) -> Spectrum:
    global _worker_cache
    if i in _worker_cache:
        _worker_cache.move_to_end(i)
        return _worker_cache[i]
    lo, hi = int(_shm_off_view[i]), int(_shm_off_view[i + 1])
    mz = _shm_mz_view[lo:hi].copy()
    intensities = _shm_int_view[lo:hi].copy()
    spec = Spectrum(mz=mz, intensities=intensities,
                    metadata={"precursor_mz": float(_worker_pmzs[i])})
    if len(_worker_cache) >= _CACHE_MAX:
        _worker_cache.popitem(last=False)
    _worker_cache[i] = spec
    return spec


def _build_fingerprints(
    shm_mz_view: np.ndarray, shm_int_view: np.ndarray,
    shm_off_view: np.ndarray, n: int,
    bin_width: float = 0.1, mz_max: float = 2000.0,
) -> sparse.csr_matrix:
    n_bins = int(mz_max / bin_width) + 1
    rows, cols, vals = [], [], []
    for i in range(n):
        lo, hi = int(shm_off_view[i]), int(shm_off_view[i + 1])
        if lo == hi:
            continue
        mz = shm_mz_view[lo:hi]
        ints = shm_int_view[lo:hi]
        bins = np.clip((mz / bin_width).astype(np.int32), 0, n_bins - 1)
        rows.extend([i] * len(bins))
        cols.extend(bins.tolist())
        vals.extend(ints.tolist())
    fp = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n_bins), dtype=np.float64)
    # L2-normalize rows
    norms = sparse.linalg.norm(fp, axis=1)
    norms[norms == 0] = 1.0
    fp = fp.multiply(1.0 / norms[:, np.newaxis]).tocsr()
    return fp


def _fingerprint_cosine(i: int, j: int) -> float:
    return float(_worker_fp[i].dot(_worker_fp[j].T).toarray()[0, 0])


def _score_row_shm(args: tuple) -> list[tuple[int, int, float, int]]:
    i, cosine_threshold, min_matched_peaks, mz_tolerance, max_mz_diff = args
    pair = ModifiedCosine(tolerance=mz_tolerance).pair
    mz_i = float(_worker_pmzs[i])
    n = _worker_n
    results = []
    spec_i = _get_spectrum(i)
    if max_mz_diff > 0:
        j_hi = int(np.searchsorted(_worker_pmzs, mz_i + max_mz_diff, side='right'))
    else:
        j_hi = n
    for j in range(i + 1, j_hi):
        if _worker_fp is not None:
            fp_score = _fingerprint_cosine(i, j)
            if fp_score < _worker_fp_thresh:
                continue
        r = pair(spec_i, _get_spectrum(j))
        if r['score'] >= cosine_threshold and r['matches'] >= min_matched_peaks:
            results.append((i, j, float(r['score']), int(r['matches'])))
    return results


# =============================================================================
# 1. DATA STRUCTURES
# =============================================================================

@dataclass
class Annotation:
    """A candidate annotation for an unknown spectrum."""
    compound_name: str
    smiles: Optional[str] = None
    inchi: Optional[str] = None
    molecular_formula: Optional[str] = None
    adduct: Optional[str] = None
    score: float = 0.0
    matched_peaks: int = 0
    source: str = "unknown"          # "direct_match" or "propagation"
    propagation_depth: int = 0       # 0 = direct, 1 = 1-hop neighbor, etc.
    network_node_id: Optional[str] = None
    mass_shift: float = 0.0          # precursor mass difference (for analogs)


# =============================================================================
# 2. MOLECULAR NETWORK BUILDER
# =============================================================================

def _serialize_node_attrs(spec: Spectrum, node_id: str) -> dict:
    """Build a JSON-serializable node dict (node-link format)."""
    return {
        "id": node_id,
        "spectrum": _serialize_value(spec),
        "node_id": node_id,
        "precursor_mz": _serialize_value(spec.get("precursor_mz")),
        "compound_name": spec.get("compound_name") or spec.get("name"),
        "smiles": spec.get("smiles"),
        "inchi": spec.get("inchi"),
        "molecular_formula": spec.get("molecular_formula") or spec.get("formula"),
        "adduct": spec.get("adduct"),
    }


def _heap_push(
    heaps: list[list], node_idx: int, score: float,
    other_idx: int, matches: int, top_k: int,
):
    """Push (score, other_idx, matches) into a min-heap, evict if over top_k."""
    h = heaps[node_idx]
    entry = (score, other_idx, matches)
    if len(h) < top_k:
        heapq.heappush(h, entry)
    elif score > h[0][0]:
        heapq.heapreplace(h, entry)


def build_molecular_network(
    spectra: list[Spectrum],
    cosine_threshold: float = 0.7,
    min_matched_peaks: int = 6,
    top_k: int = 10,
    mz_tolerance: float = 0.02,
    max_mz_diff: float = 500.0,
    n_workers: int = None,
    path: str = "",
    fingerprint_threshold: float = 0.1,
) -> nx.Graph:
    n = len(spectra)
    if n == 0:
        return nx.Graph()

    node_ids = [s.get("spectrum_id") or f"spec_{i}" for i, s in enumerate(spectra)]
    precursor_mzs = np.array([float(s.get("precursor_mz") or 0.0) for s in spectra])

    # Sort by precursor m/z for bisect-based window pruning
    order = np.argsort(precursor_mzs)
    spectra = [spectra[k] for k in order]
    node_ids = [node_ids[k] for k in order]
    precursor_mzs = precursor_mzs[order]

    # Build JSON nodes
    json_nodes = [_serialize_node_attrs(s, nid) for s, nid in zip(spectra, node_ids)]

    # Pack peak arrays into shared memory
    shm_mz, shm_int, shm_off, offsets = _pack_into_shared_memory(spectra)
    total_peaks = int(offsets[-1])
    del spectra

    n_workers = n_workers or os.cpu_count()
    print(f"Computing pairwise similarities for {n} spectra on {n_workers} workers...")

    # Build sparse fingerprints for pre-filtering
    mz_view = np.ndarray((total_peaks,), dtype=np.float64, buffer=shm_mz.buf)
    int_view = np.ndarray((total_peaks,), dtype=np.float64, buffer=shm_int.buf)
    off_view = np.ndarray((n + 1,), dtype=np.int64, buffer=shm_off.buf)
    fp = _build_fingerprints(mz_view, int_view, off_view, n)

    fp_data = np.ascontiguousarray(fp.data, dtype=np.float64)
    fp_indices = np.ascontiguousarray(fp.indices, dtype=np.int32)
    fp_indptr = np.ascontiguousarray(fp.indptr, dtype=np.int64)
    shm_fp_d = shared_memory.SharedMemory(create=True, size=max(8, fp_data.nbytes))
    shm_fp_i = shared_memory.SharedMemory(create=True, size=max(8, fp_indices.nbytes))
    shm_fp_p = shared_memory.SharedMemory(create=True, size=max(8, fp_indptr.nbytes))
    np.ndarray(fp_data.shape, dtype=fp_data.dtype, buffer=shm_fp_d.buf)[:] = fp_data
    np.ndarray(fp_indices.shape, dtype=fp_indices.dtype, buffer=shm_fp_i.buf)[:] = fp_indices
    np.ndarray(fp_indptr.shape, dtype=fp_indptr.dtype, buffer=shm_fp_p.buf)[:] = fp_indptr
    fp_shm_names = (shm_fp_d.name, shm_fp_i.name, shm_fp_p.name)
    fp_shapes = (len(fp_data), fp.shape[0], fp.shape[1])
    del fp, fp_data, fp_indices, fp_indptr, mz_view, int_view, off_view

    tasks = [
        (i, cosine_threshold, min_matched_peaks, mz_tolerance, max_mz_diff)
        for i in range(n)
    ]
    chunksize = max(1, n // (n_workers * 4))

    # Per-node min-heaps for top-K accumulation
    heaps: list[list] = [[] for _ in range(n)]

    ctx = _mp_context()
    try:
        with ctx.Pool(
            processes=n_workers,
            initializer=_init_worker_shm,
            initargs=(shm_mz.name, shm_int.name, shm_off.name,
                      total_peaks, n, precursor_mzs.tolist(),
                      fp_shm_names, fp_shapes, fingerprint_threshold),
        ) as pool:
            for row in tqdm(
                pool.imap(_score_row_shm, tasks, chunksize=chunksize),
                total=n,
            ):
                for i, j, score, matches in row:
                    _heap_push(heaps, i, score, j, matches, top_k)
                    _heap_push(heaps, j, score, i, matches, top_k)
    finally:
        for shm in (shm_mz, shm_int, shm_off, shm_fp_d, shm_fp_i, shm_fp_p):
            shm.close()
            shm.unlink()

    # Collect unique edges from heaps
    seen_edges: set[tuple[int, int]] = set()
    json_links = []
    for idx in range(n):
        for score, other, matches in heaps[idx]:
            edge_key = (min(idx, other), max(idx, other))
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                json_links.append({
                    "source": node_ids[edge_key[0]],
                    "target": node_ids[edge_key[1]],
                    "weight": score,
                    "matched_peaks": matches,
                })

    json_data = {
        "directed": False,
        "multigraph": False,
        "graph": {},
        "nodes": json_nodes,
        "edges": json_links,
    }

    n_annotated = sum(1 for nd in json_nodes if nd.get("compound_name"))
    print(f"Network: {n} nodes, {len(json_links)} edges, {n_annotated} annotated nodes")

    if path:
        with open(path, "w") as f:
            json.dump(json_data, f)

    return json_to_graph(json_data)


def build_molecular_network_vanilla(
    spectra: list[Spectrum],
    cosine_threshold: float = 0.7,
    min_matched_peaks: int = 6,
    top_k: int = 10,
    mz_tolerance: float = 0.02,
    max_mz_diff: float = 500.0,
    path: str = "",
) -> nx.Graph:
    """Single-process network builder — no shared memory, no fingerprint filter.

    Use this for small datasets where multiprocessing overhead is not worth it
    and where you want exact results (no fingerprint pre-filtering).
    """
    n = len(spectra)
    if n == 0:
        return nx.Graph()

    node_ids = [s.get("spectrum_id") or f"spec_{i}" for i, s in enumerate(spectra)]
    precursor_mzs = [float(s.get("precursor_mz") or 0.0) for s in spectra]

    json_nodes = [_serialize_node_attrs(s, nid) for s, nid in zip(spectra, node_ids)]

    heaps: list[list] = [[] for _ in range(n)]
    similarity = ModifiedCosine(tolerance=mz_tolerance)

    print(f"Computing pairwise similarities for {n} spectra (single process)...")
    for i in tqdm(range(n)):
        mz_i = precursor_mzs[i]
        for j in range(i + 1, n):
            if max_mz_diff > 0 and abs(mz_i - precursor_mzs[j]) > max_mz_diff:
                continue
            r = similarity.pair(spectra[i], spectra[j])
            score = float(r["score"])
            matches = int(r["matches"])
            if score >= cosine_threshold and matches >= min_matched_peaks:
                _heap_push(heaps, i, score, j, matches, top_k)
                _heap_push(heaps, j, score, i, matches, top_k)

    seen_edges: set[tuple[int, int]] = set()
    json_links = []
    for idx in range(n):
        for score, other, matches in heaps[idx]:
            edge_key = (min(idx, other), max(idx, other))
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                json_links.append({
                    "source": node_ids[edge_key[0]],
                    "target": node_ids[edge_key[1]],
                    "weight": score,
                    "matched_peaks": matches,
                })

    json_data = {
        "directed": False,
        "multigraph": False,
        "graph": {},
        "nodes": json_nodes,
        "edges": json_links,
    }

    n_annotated = sum(1 for nd in json_nodes if nd.get("compound_name"))
    print(f"Network: {n} nodes, {len(json_links)} edges, {n_annotated} annotated nodes")

    if path:
        with open(path, "w") as f:
            json.dump(json_data, f)

    return json_to_graph(json_data)
