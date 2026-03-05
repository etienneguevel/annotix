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

import pandas as pd
import networkx as nx
from dataclasses import dataclass
from typing import Optional
from matchms import Spectrum
from matchms.similarity import ModifiedCosine

from .analog_inference import infer_analog_structure, StructureCandidate


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
