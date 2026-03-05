"""
Spectral Annotation Pipeline
=============================
Simple approach to annotate unknown MS/MS spectra using:
1. Direct spectral matching (modified cosine similarity)
2. Annotation propagation through a molecular network

Dependencies:
    pip install matchms networkx pandas numpy

Inputs:
    - A molecular network: a NetworkX graph where nodes carry MS/MS spectra
      and (some) annotations from a reference library (e.g., GNPS)
    - One or more unknown MS/MS spectra to annotate

Output:
    - Ranked candidate annotations with confidence scores and provenance
"""

import heapq
import multiprocessing as mp
from multiprocessing import shared_memory
import os
import sys
from collections import OrderedDict

import pandas as pd
import networkx as nx
import numpy as np
from dataclasses import dataclass
from typing import Optional
from matchms import Spectrum
from matchms.similarity import ModifiedCosine
from tqdm import tqdm
import json

from scipy import sparse

from store import json_to_graph, _serialize_value
from analog_inference import infer_analog_structure, StructureCandidate


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


# =============================================================================
# 3. DIRECT SPECTRAL MATCHING
# =============================================================================

def direct_match(
    query: Spectrum,
    network: nx.Graph,
    mz_tolerance: float = 0.02,
    score_threshold: float = 0.7,
    min_matched_peaks: int = 6,
    top_n: int = 10,
) -> list[Annotation]:
    """
    Compare the unknown spectrum against all spectra in the network.

    Returns ranked list of direct match annotations.

    Parameters
    ----------
    query : matchms.Spectrum
        The unknown spectrum to annotate.
    network : nx.Graph
        The molecular network (nodes carry spectra).
    score_threshold : float
        Minimum cosine score for a valid match.
    min_matched_peaks : int
        Minimum shared fragment peaks.
    top_n : int
        Return top N matches.

    Returns
    -------
    list of Annotation
    """
    similarity_function = ModifiedCosine(tolerance=mz_tolerance)
    candidates = []

    query_mz = query.get("precursor_mz") or 0.0

    for node_id, data in network.nodes(data=True):
        ref_spectrum = data["spectrum"]
        try:
            result = similarity_function.pair(query, ref_spectrum)
            # ModifiedCosine returns a structured numpy array with fields 'score' and 'matches'
            score = float(result['score'])
            n_matched = int(result['matches'])
        except Exception as err:
            print(err)
            continue

        if score >= score_threshold and n_matched >= min_matched_peaks:
            ref_mz = data.get("precursor_mz") or 0.0
            candidates.append(Annotation(
                compound_name=data.get("compound_name") or "Unknown",
                smiles=data.get("smiles"),
                inchi=data.get("inchi"),
                molecular_formula=data.get("molecular_formula"),
                adduct=data.get("adduct"),
                score=score,
                matched_peaks=n_matched,
                source="direct_match",
                propagation_depth=0,
                network_node_id=node_id,
                mass_shift=abs(query_mz - ref_mz) if query_mz and ref_mz else 0.0,
            ))

    # Rank by score descending
    candidates.sort(key=lambda a: (-a.score, -a.matched_peaks))
    return candidates[:top_n]


# =============================================================================
# 4. ANNOTATION PROPAGATION
# =============================================================================

def propagate_annotations(
    query: Spectrum,
    network: nx.Graph,
    direct_matches: list[Annotation],
    mz_tolerance: float = 0.02,
    score_threshold: float = 0.5,
    min_matched_peaks: int = 4,
    max_depth: int = 2,
    decay_factor: float = 0.7,
    top_n: int = 10,
) -> list[Annotation]:
    """
    Propagate annotations through the molecular network.

    Algorithm
    ---------
    1. Find the best-matching nodes in the network for the query
       (even below the direct match threshold — these are "anchor" nodes).
    2. From each anchor, walk the network up to `max_depth` hops.
    3. For each reachable annotated node, compute a propagation score:

        propagation_score = anchor_similarity × Π(edge_weights along path) × decay^depth

       This captures:
       - How similar the query is to its entry point in the network
       - How strongly connected the path is (edge weights = cosine similarities)
       - A depth penalty (further = less confident)

    4. Aggregate scores per unique compound (same SMILES or name) and rank.

    Parameters
    ----------
    query : Spectrum
        The unknown spectrum.
    network : nx.Graph
        The molecular network.
    direct_matches : list of Annotation
        Already found direct matches (to avoid duplicates).
    score_threshold : float
        Lower threshold for anchor similarity (more permissive than direct).
    min_matched_peaks : int
        Lower threshold for anchoring.
    max_depth : int
        Maximum number of hops from anchor node.
    decay_factor : float
        Score penalty per hop (0-1). 0.7 means 70% retained per hop.
    top_n : int
        Return top N propagated annotations.

    Returns
    -------
    list of Annotation
    """
    similarity_function = ModifiedCosine(tolerance=mz_tolerance)
    query_mz = query.get("precursor_mz") or 0.0

    # --- Step 1: Find anchor nodes (best matches in the network) ---
    anchors = []
    for node_id, data in network.nodes(data=True):
        ref_spectrum = data["spectrum"]

        try:
            result = similarity_function.pair(query, ref_spectrum)
            # ModifiedCosine returns a structured numpy array with fields 'score' and 'matches'
            score = float(result['score'])
            n_matched = int(result['matches'])
        except Exception as err:
            print(err)
            continue

        if score >= score_threshold and n_matched >= min_matched_peaks:
            anchors.append((node_id, score))
            print(node_id, score)

    if not anchors:
        return []

    # Sort anchors by similarity (best first)
    anchors.sort(key=lambda x: -x[1])

    # --- Step 2: BFS from each anchor, collect annotated nodes ---
    direct_match_nodes = {a.network_node_id for a in direct_matches}
    propagated: dict[str, Annotation] = {}  # keyed by compound identifier

    for anchor_id, anchor_score in anchors:
        # BFS with depth tracking
        visited = {anchor_id}
        frontier = [(anchor_id, 0, 1.0)]  # (node, depth, cumulative_edge_weight)

        while frontier:
            current_id, depth, path_weight = frontier.pop(0)

            if depth > max_depth:
                continue

            # Check if this node is annotated
            node_data = network.nodes[current_id]
            compound_name = node_data.get("compound_name")

            if compound_name and depth > 0:  # depth > 0: skip the anchor itself
                # Compute propagation score
                prop_score = anchor_score * path_weight * (decay_factor ** depth)

                ref_mz = node_data.get("precursor_mz") or 0.0
                mass_shift = abs(query_mz - ref_mz) if query_mz and ref_mz else 0.0

                # Compound key: use SMILES if available, else name
                key = node_data.get("smiles") or compound_name

                # Keep the best score per compound
                if key not in propagated or prop_score > propagated[key].score:
                    propagated[key] = Annotation(
                        compound_name=compound_name,
                        smiles=node_data.get("smiles"),
                        inchi=node_data.get("inchi"),
                        molecular_formula=node_data.get("molecular_formula"),
                        adduct=node_data.get("adduct"),
                        score=prop_score,
                        matched_peaks=0,  # not directly matched
                        source="propagation",
                        propagation_depth=depth,
                        network_node_id=current_id,
                        mass_shift=mass_shift,
                    )

            # Expand neighbors
            if depth < max_depth:
                for neighbor_id in network.neighbors(current_id):
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        edge_weight = network[current_id][neighbor_id].get("weight", 0.5)
                        frontier.append((
                            neighbor_id,
                            depth + 1,
                            path_weight * edge_weight,
                        ))

    # Remove compounds already found by direct matching
    for dm in direct_matches:
        key = dm.smiles or dm.compound_name
        propagated.pop(key, None)

    # Rank by propagation score
    results = sorted(propagated.values(), key=lambda a: -a.score)
    return results[:top_n]


# =============================================================================
# 5. FULL ANNOTATION PIPELINE
# =============================================================================

def annotate(
    query: Spectrum,
    network: nx.Graph,
    # Direct match params
    direct_score_threshold: float = 0.7,
    direct_min_matched: int = 6,
    # Propagation params
    prop_score_threshold: float = 0.5,
    prop_min_matched: int = 4,
    max_depth: int = 2,
    decay_factor: float = 0.7,
    # General
    mz_tolerance: float = 0.02,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Full annotation pipeline: direct matching + propagation.

    Returns a DataFrame with ranked candidate annotations.
    """
    # --- Direct matching ---
    direct = direct_match(
        query, network,
        mz_tolerance=mz_tolerance,
        score_threshold=direct_score_threshold,
        min_matched_peaks=direct_min_matched,
        top_n=top_n,
    )

    # --- Propagation ---
    propagated = propagate_annotations(
        query, network, direct,
        mz_tolerance=mz_tolerance,
        score_threshold=prop_score_threshold,
        min_matched_peaks=prop_min_matched,
        max_depth=max_depth,
        decay_factor=decay_factor,
        top_n=top_n,
    )

    # --- Combine and format ---
    all_annotations = direct + propagated

    if not all_annotations:
        return pd.DataFrame(columns=[
            "rank", "compound_name", "smiles", "inchi", "molecular_formula",
            "adduct", "score", "matched_peaks", "source", "propagation_depth",
            "mass_shift_da", "network_node_id", "confidence_level",
        ])

    rows = []
    for i, ann in enumerate(all_annotations):
        rows.append({
            "rank": i + 1,
            "compound_name": ann.compound_name,
            "smiles": ann.smiles,
            "inchi": ann.inchi,
            "molecular_formula": ann.molecular_formula,
            "adduct": ann.adduct,
            "score": round(ann.score, 4),
            "matched_peaks": ann.matched_peaks,
            "source": ann.source,
            "propagation_depth": ann.propagation_depth,
            "mass_shift_da": round(ann.mass_shift, 4),
            "network_node_id": ann.network_node_id,
            "confidence_level": _assign_msi_level(ann),
        })

    return pd.DataFrame(rows)


def _assign_msi_level(ann: Annotation) -> str:
    """
    Assign Metabolomics Standards Initiative (MSI) confidence level.

    Level 1: Confirmed (requires RT + MS/MS match to authentic standard)
    Level 2: Probable (high-confidence spectral library match)
    Level 3: Putative class (propagated or low-confidence match)
    Level 4: Unknown
    """
    if ann.source == "direct_match" and ann.score >= 0.85 and ann.matched_peaks >= 8:
        return "MSI-2 (probable)"
    elif ann.source == "direct_match" and ann.score >= 0.7:
        return "MSI-2 (probable, lower confidence)"
    elif ann.source == "propagation" and ann.propagation_depth <= 1:
        return "MSI-3 (putative, 1-hop neighbor)"
    elif ann.source == "propagation":
        return "MSI-3 (putative, multi-hop)"
    else:
        return "MSI-4 (unknown)"


# =============================================================================
# 6. BATCH ANNOTATION
# =============================================================================

def annotate_batch(
    queries: list[Spectrum],
    network: nx.Graph,
    **kwargs,
) -> pd.DataFrame:
    """
    Annotate multiple unknown spectra.

    Returns a combined DataFrame with a query_id column.
    """
    all_results = []

    for i, query in enumerate(queries):
        query_id = query.get("spectrum_id") or f"query_{i}"
        print(f"Annotating {query_id} ({i+1}/{len(queries)})...")

        df = annotate(query, network, **kwargs)
        df.insert(0, "query_id", query_id)
        all_results.append(df)

    if all_results:
        return pd.concat(all_results, ignore_index=True) #some NA or empty dataframe so a warning occurs
    return pd.DataFrame()


# =============================================================================
# 7. EXPORT
# =============================================================================

def export_results(df: pd.DataFrame, output_path: str, format: str = "tsv"):
    """Export annotation results to file."""
    if format == "tsv":
        df.to_csv(output_path, sep="\t", index=False)
    elif format == "csv":
        df.to_csv(output_path, index=False)
    elif format == "xlsx":
        df.to_excel(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    print(f"Results exported to {output_path}")


def infer_structure(
    query: Spectrum,
    result_df: pd.DataFrame,
    network: nx.Graph,
    mz_tolerance: float = 0.02,
    delta_mass_tolerance: float = 0.02,
    max_candidates_per_mod: int = 10,
    top_n: int = 5,
) -> list[StructureCandidate]:
    """
    Attempt to infer the modified structure of an unknown analog.

    This is the third step after annotate() + select_annotation().
    It should be called when select_annotation() returns type="analog"
    or type="putative" — i.e. when the unknown is not directly identified
    but a close neighbor is known.

    Pipeline (delegated to analog_inference.py)
    -------------------------------------------
    1. Resolve the precursor Δm to a biochemical formula delta (+CH2, +O, …).
    2. Fragment differential analysis: conserved vs. shifted peaks.
    3. Enumerate structurally modified SMILES from the reference SMILES (RDKit).
    4. Simplified in silico fragmentation by bond cleavage.
    5. Score predicted fragments against the query spectrum.

    Parameters
    ----------
    query : Spectrum
        The unknown spectrum (must carry precursor_mz).
    result_df : pd.DataFrame
        Output of annotate(). The best candidate (row 0) supplies the
        reference SMILES and network node id.
    network : nx.Graph
        The molecular network (nodes carry the reference Spectrum objects).
    mz_tolerance : float
        Fragment matching tolerance in Da.
    delta_mass_tolerance : float
        Tolerance for Δm → modification lookup.
    max_candidates_per_mod : int
        Maximum structural variants explored per modification type.
    top_n : int
        Number of top candidates to return.

    Returns
    -------
    List of StructureCandidate ranked by fragmentation_score.
    Empty list if: RDKit is unavailable, the best candidate has no SMILES,
    the Δm is not resolved, or the result_df is empty.
    """
    if result_df.empty:
        return []

    best = result_df.iloc[0]
    reference_smiles = best.get("smiles")
    node_id = best.get("network_node_id")

    if not reference_smiles or not node_id:
        return []

    node_data = network.nodes.get(node_id)
    if node_data is None:
        return []

    reference_spectrum = node_data.get("spectrum")
    if reference_spectrum is None:
        return []

    return infer_analog_structure(
        query=query,
        reference_smiles=reference_smiles,
        reference_spectrum=reference_spectrum,
        mz_tolerance=mz_tolerance,
        delta_mass_tolerance=delta_mass_tolerance,
        max_candidates_per_mod=max_candidates_per_mod,
        top_n=top_n,
    )


def select_annotation(result_df: pd.DataFrame) -> dict | None:
    if result_df.empty:
        return None  # Scenario 3: truly unknown

    best = result_df.iloc[0]  # already ranked by score

    # Scenario 1: confident direct match, same compound
    if best["source"] == "direct_match" and best["score"] >= 0.7 and best["mass_shift_da"] < 0.02:
        return {"annotation": best["compound_name"], "confidence": best["confidence_level"], "type": "identified"}

    # Scenario 2a: direct match but mass-shifted → structural analog
    if best["source"] == "direct_match" and best["mass_shift_da"] >= 0.02:
        return {"annotation": f"analog of {best['compound_name']}", "confidence": "MSI-3", "type": "analog"}

    # Scenario 2b: propagated → putative class member
    if best["source"] == "propagation" and best["propagation_depth"] <= 1 and best["score"] >= 0.3:
        return {"annotation": f"putative: {best['compound_name']}", "confidence": "MSI-3", "type": "putative"}

    return None  # not confident enough
