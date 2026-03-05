"""
Analog Structure Inference
==========================
Infers the structure of an unknown MS/MS spectrum that is a structural analog
of a known compound, using:

  1. Δm lookup          → map precursor mass shift to biochemical formula delta
  2. Fragment diff      → classify peaks as conserved, shifted, novel, lost
  3. Structure enum     → apply formula delta to reference SMILES (RDKit)
  4. In silico frag     → predict fragment masses by bond cleavage (simplified)
  5. Spectral scoring   → match predicted fragments to the observed spectrum

Dependencies:
    pip install rdkit matchms numpy
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from matchms import Spectrum

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn(
        "RDKit not installed — structure enumeration unavailable. "
        "Install with: pip install rdkit",
        ImportWarning,
        stacklevel=2,
    )

# =============================================================================
# 1. DELTA MASS LOOKUP
# =============================================================================

# (exact_monoisotopic_mass_da, formula_delta_tag, human_readable_name)
# formula_delta_tag is used as a key into _ENUMERATION_DISPATCH below.
_DELTA_MASS_TABLE: list[tuple[float, str, str]] = [
    (2.0157,   "+H2",      "reduction / hydrogenation"),
    (2.0157,   "-H2",      "oxidation / dehydrogenation"),
    (14.0157,  "+CH2",     "chain elongation or N-methylation"),
    (15.9949,  "+O",       "hydroxylation / oxidation"),
    (18.0106,  "-H2O",     "dehydration (ring closure or elimination)"),
    (27.9949,  "+CO",      "carbonylation (aldehyde / ketone insertion)"),
    (28.0313,  "+C2H4",    "chain elongation +2C (homologation)"),
    (42.0106,  "+C2H2O",   "acetylation"),
    (44.0262,  "+CO2",     "carboxylation"),
    (79.9663,  "+HPO3",    "phosphorylation"),
    (162.0528, "+Hex",     "hexosylation (O-glycoside)"),
    (146.0579, "+dHex",    "deoxyhexosylation"),
]

_PROTON = 1.007276  # Da, for [M+H]+ ions


def resolve_delta_mass(
    delta_da: float,
    tolerance_da: float = 0.02,
) -> list[dict]:
    """
    Map the signed precursor mass shift to known biochemical modifications.

    Parameters
    ----------
    delta_da : float
        Signed Δm = query_precursor_mz - reference_precursor_mz.
        Positive → unknown is heavier; negative → unknown is lighter.
    tolerance_da : float
        Matching tolerance in Da.

    Returns
    -------
    List of dicts (may be empty), each with keys:
        exact_mass_da, formula_delta, name, direction, mass_error_da.
    """
    abs_delta = abs(delta_da)
    direction = "gain" if delta_da >= 0 else "loss"
    hits = []
    for exact, formula, name in _DELTA_MASS_TABLE:
        print(formula, delta_da, abs(abs_delta - exact), tolerance_da)
        if abs(abs_delta - exact) <= tolerance_da:
            hits.append({
                "exact_mass_da": exact,
                "formula_delta": formula,
                "name": name,
                "direction": direction,
                "mass_error_da": round(abs_delta - exact, 5),
            })
    return hits


# =============================================================================
# 2. FRAGMENT DIFFERENTIAL ANALYSIS
# =============================================================================

@dataclass
class FragmentDiff:
    """
    Per-peak classification between an unknown and a reference spectrum.

    conserved_mzs : peaks present in both at the same m/z
                    → modification is NOT in this substructure
    shifted_mzs   : peaks in the unknown shifted by Δm vs the reference
                    → modification IS in this substructure
    novel_mzs     : peaks only in the unknown (arise from the modified region)
    lost_mzs      : peaks only in the reference (lost upon modification)
    localization_score : n_shifted / (n_conserved + n_shifted)
                    → 0.0 means the modification is not reflected in the
                      fragment ions; 1.0 means all fragments carry the shift
    """
    conserved_mzs: np.ndarray
    shifted_mzs: np.ndarray
    novel_mzs: np.ndarray
    lost_mzs: np.ndarray
    delta_mz: float
    n_conserved: int
    n_shifted: int
    localization_score: float


def fragment_differential(
    query: Spectrum,
    reference: Spectrum,
    delta_mz: float,
    mz_tolerance: float = 0.02,
) -> FragmentDiff:
    """
    Classify each fragment peak of the unknown as conserved, shifted, novel,
    or lost relative to the reference.

    Parameters
    ----------
    query : Spectrum
        The unknown spectrum.
    reference : Spectrum
        The best-matching reference spectrum from the network.
    delta_mz : float
        Signed Δm = query_mz - ref_mz  (same sign convention as resolve_delta_mass).
    mz_tolerance : float
        Per-peak matching tolerance in Da.
    """
    q_mz = np.asarray(query.mz, dtype=np.float64)
    r_mz = np.asarray(reference.mz, dtype=np.float64)

    conserved, shifted, novel = [], [], []
    matched_ref: set[int] = set()

    for qm in q_mz:
        # --- Conserved: same peak in the reference ---
        diffs_same = np.abs(r_mz - qm)
        idx = int(np.argmin(diffs_same))
        if diffs_same[idx] <= mz_tolerance:
            conserved.append(qm)
            matched_ref.add(idx)
            continue

        # --- Shifted: reference peak offset by delta_mz ---
        # q_frag = r_frag + delta  =>  r_frag = q_frag - delta
        diffs_shifted = np.abs(r_mz - (qm - delta_mz))
        idx2 = int(np.argmin(diffs_shifted))
        if diffs_shifted[idx2] <= mz_tolerance:
            shifted.append(qm)
            matched_ref.add(idx2)
            continue

        novel.append(qm)

    lost = [float(r_mz[i]) for i in range(len(r_mz)) if i not in matched_ref]
    n_c, n_s = len(conserved), len(shifted)
    loc_score = n_s / (n_c + n_s) if (n_c + n_s) > 0 else 0.0

    return FragmentDiff(
        conserved_mzs=np.array(conserved),
        shifted_mzs=np.array(shifted),
        novel_mzs=np.array(novel),
        lost_mzs=np.array(lost),
        delta_mz=delta_mz,
        n_conserved=n_c,
        n_shifted=n_s,
        localization_score=loc_score,
    )


# =============================================================================
# 3. STRUCTURE ENUMERATION  (RDKit required)
# =============================================================================

def _require_rdkit():
    if not RDKIT_AVAILABLE:
        raise ImportError(
            "RDKit is required for structure enumeration. "
            "Install with: pip install rdkit"
        )


def _deduplicate(mols: list) -> list:
    """Remove duplicate molecules by canonical SMILES."""
    seen: dict[str, object] = {}
    for mol in mols:
        try:
            smi = Chem.MolToSmiles(mol)
            if smi and smi not in seen:
                seen[smi] = mol
        except Exception:
            pass
    return list(seen.values())


def _enumerate_plus_ch2(mol) -> list:
    """
    +CH2 (+14.016 Da): extend each terminal carbon by one CH2.

    Only terminal heavy-atom carbons (degree 1 in the C/heteroatom graph)
    are extended, which is the most common biochemical homologation.
    """
    results = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        heavy_nb = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
        if len(heavy_nb) != 1:
            continue  # terminal only
        rw = Chem.RWMol(Chem.Mol(mol))
        new_c = rw.AddAtom(Chem.Atom(6))
        rw.AddBond(atom.GetIdx(), new_c, Chem.BondType.SINGLE)
        try:
            Chem.SanitizeMol(rw)
            results.append(rw.GetMol())
        except Exception:
            pass
    return results


def _enumerate_plus_o(mol) -> list:
    """
    +O (+15.995 Da): add an OH group to each aliphatic CH carbon.

    Models hydroxylation (e.g. CYP450-mediated oxidation).
    """
    results = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6 or atom.GetIsAromatic():
            continue
        if atom.GetTotalNumHs() < 1:
            continue
        rw = Chem.RWMol(Chem.Mol(mol))
        new_o = rw.AddAtom(Chem.Atom(8))
        rw.AddBond(atom.GetIdx(), new_o, Chem.BondType.SINGLE)
        try:
            Chem.SanitizeMol(rw)
            results.append(rw.GetMol())
        except Exception:
            pass
    return results


def _enumerate_minus_h2o(mol) -> list:
    """
    -H2O (-18.011 Da): beta-elimination of water from each C-OH group.

    Uses a SMARTS reaction:  [C:1][C:2][OH] >> [C:1]=[C:2]
    The hydroxyl oxygen and its H are removed; the adjacent C-C bond becomes C=C.
    """
    rxn = AllChem.ReactionFromSmarts("[C:1][C:2][OH]>>[C:1]=[C:2]")
    results = []
    for products_set in rxn.RunReactants((mol,)):
        for prod in products_set:
            try:
                Chem.SanitizeMol(prod)
                results.append(prod)
            except Exception:
                pass
    return results


def _enumerate_plus_co(mol) -> list:
    """
    +CO (+27.995 Da): append a formyl group (CHO) to each terminal carbon.

    Models carbonylation or aldehyde extension.
    """
    results = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        heavy_nb = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
        if len(heavy_nb) != 1:
            continue
        rw = Chem.RWMol(Chem.Mol(mol))
        c_co = rw.AddAtom(Chem.Atom(6))
        o_co = rw.AddAtom(Chem.Atom(8))
        rw.AddBond(atom.GetIdx(), c_co, Chem.BondType.SINGLE)
        rw.AddBond(c_co, o_co, Chem.BondType.DOUBLE)
        try:
            Chem.SanitizeMol(rw)
            results.append(rw.GetMol())
        except Exception:
            pass
    return results


def _enumerate_plus_c2h4(mol) -> list:
    """
    +C2H4 (+28.031 Da): two successive +CH2 extensions.

    Models elongation by two carbons (e.g. fatty acid chain extension).
    """
    results = []
    for m in _enumerate_plus_ch2(mol):
        results.extend(_enumerate_plus_ch2(m))
    return results


def _enumerate_plus_acetyl(mol) -> list:
    """
    +C2H2O / acetylation (+42.011 Da): add CH3C(=O)- to each OH or NH group.
    """
    rxn_oh = AllChem.ReactionFromSmarts("[OH:1]>>[O:1]C(=O)C")
    rxn_nh = AllChem.ReactionFromSmarts("[NH:1]>>[N:1]C(=O)C")
    results = []
    for rxn in (rxn_oh, rxn_nh):
        for products_set in rxn.RunReactants((mol,)):
            for prod in products_set:
                try:
                    Chem.SanitizeMol(prod)
                    results.append(prod)
                except Exception:
                    pass
    return results


# Dispatch table: formula_delta_tag → enumeration function
_ENUMERATION_DISPATCH: dict[str, object] = {
    "+CH2":    _enumerate_plus_ch2,
    "+O":      _enumerate_plus_o,
    "-H2O":    _enumerate_minus_h2o,
    "+CO":     _enumerate_plus_co,
    "+C2H4":   _enumerate_plus_c2h4,
    "+C2H2O":  _enumerate_plus_acetyl,
}


def enumerate_modified_structures(
    smiles: str,
    formula_delta: str,
    max_candidates: int = 20,
) -> list[str]:
    """
    Enumerate plausible structural modifications of a known compound.

    Parameters
    ----------
    smiles : str
        SMILES of the reference (known) compound.
    formula_delta : str
        Tag from resolve_delta_mass (e.g. "+CH2", "+O", "-H2O").
    max_candidates : int
        Maximum number of unique candidate SMILES returned.

    Returns
    -------
    List of canonical SMILES strings.
    Empty list if the modification is not supported or RDKit is unavailable.
    """
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    fn = _ENUMERATION_DISPATCH.get(formula_delta)
    if fn is None:
        return []
    candidates = _deduplicate(fn(mol))[:max_candidates]
    return [Chem.MolToSmiles(m) for m in candidates]


# =============================================================================
# 4. SIMPLIFIED IN SILICO FRAGMENTATION
# =============================================================================

def _predict_fragments(mol, max_fragments: int = 150) -> list[float]:
    """
    Predict fragment m/z values by cleaving each acyclic single bond.

    For each cleavage:
    - Two substructure fragments are generated.
    - Each is recorded as a protonated ion [frag+H]+ (m/z = mass + 1.00728).

    Additionally, common neutral losses from the precursor [M+H]+ are included:
    -H2O, -NH3, -CO, -CO2.

    Limitations
    -----------
    This is a first-order approximation. Real MS/MS fragmentation involves
    rearrangements, ring openings, and multi-step mechanisms not captured here.
    Use CFM-ID or MetFrag for production-grade in silico fragmentation.

    Parameters
    ----------
    mol : RDKit Mol
    max_fragments : int
        Hard cap on the number of predicted m/z values.

    Returns
    -------
    Sorted list of unique predicted fragment m/z values (rounded to 4 dp).
    """
    ring_info = mol.GetRingInfo()
    ring_bond_set: set[int] = set()
    for ring in ring_info.BondRings():
        ring_bond_set.update(ring)

    precursor_mass = Descriptors.ExactMolWt(mol)
    prec_mh = precursor_mass + _PROTON

    masses: set[float] = set()

    # Bond cleavage fragments
    for bond in mol.GetBonds():
        if len(masses) >= max_fragments:
            break
        if bond.GetIdx() in ring_bond_set:
            continue
        if bond.GetBondTypeAsDouble() != 1.0:
            continue
        rw = Chem.RWMol(Chem.Mol(mol))
        rw.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        frags = Chem.GetMolFrags(rw.GetMol(), asMols=True, sanitizeFrags=True)
        if len(frags) != 2:
            continue
        for frag in frags:
            try:
                masses.add(round(Descriptors.ExactMolWt(frag) + _PROTON, 4))
            except Exception:
                pass

    # Common neutral losses from [M+H]+
    for loss in [18.0106, 17.0265, 27.9949, 44.0262]:  # H2O, NH3, CO, CO2
        masses.add(round(prec_mh - loss, 4))

    return sorted(masses)


# =============================================================================
# 5. SPECTRAL SCORING
# =============================================================================

def _score_candidate(
    predicted_mzs: list[float],
    query: Spectrum,
    mz_tolerance: float = 0.02,
) -> float:
    """
    Score a candidate's predicted fragments against the observed spectrum.

    Score = Σ intensity of matched peaks / Σ total intensity.
    Intensities are normalised to [0, 1] before summation.

    Parameters
    ----------
    predicted_mzs : list of float
    query : Spectrum
    mz_tolerance : float

    Returns
    -------
    float in [0, 1].
    """
    if not predicted_mzs:
        return 0.0

    q_mz = np.asarray(query.mz, dtype=np.float64)
    q_int = np.asarray(query.intensities, dtype=np.float64)
    total = q_int.sum()
    if total <= 0:
        return 0.0
    q_int = q_int / total

    pred = np.array(sorted(predicted_mzs), dtype=np.float64)
    matched = 0.0
    for qm, qi in zip(q_mz, q_int):
        if np.abs(pred - qm).min() <= mz_tolerance:
            matched += qi
    return float(matched)


# =============================================================================
# 6. MAIN ENTRY POINT
# =============================================================================

@dataclass
class StructureCandidate:
    """
    A candidate structure inferred for an unknown analog.

    Attributes
    ----------
    smiles : str
        Canonical SMILES of the candidate.
    formula_delta : str
        Applied modification tag (e.g. "+CH2").
    modification_name : str
        Human-readable modification description.
    fragmentation_score : float
        Fraction of observed intensity explained by predicted fragments [0–1].
    localization_score : float
        Fraction of explained fragment peaks that are shifted vs. conserved [0–1].
        Higher = modification is localised in the fragment ion series.
    n_conserved_peaks : int
    n_shifted_peaks : int
    msi_level : str
    """
    smiles: str
    formula_delta: str
    modification_name: str
    fragmentation_score: float
    localization_score: float
    n_conserved_peaks: int
    n_shifted_peaks: int
    msi_level: str = "MSI-3 (putative analog)"


def infer_analog_structure(
    query: Spectrum,
    reference_smiles: str,
    reference_spectrum: Spectrum,
    mz_tolerance: float = 0.02,
    delta_mass_tolerance: float = 0.02,
    max_candidates_per_mod: int = 10,
    top_n: int = 5,
) -> list[StructureCandidate]:
    """
    Infer the structure of an unknown analog from spectral evidence.

    Pipeline
    --------
    1. Compute signed Δm from precursor masses.
    2. Resolve Δm to biochemical modifications (lookup table).
    3. Fragment differential analysis: classify peaks as conserved / shifted.
    4. For each resolved modification:
       a. Enumerate structurally modified SMILES with RDKit.
       b. Predict fragment masses by bond cleavage.
       c. Score predicted fragments against the unknown spectrum.
    5. Return top-N candidates ranked by fragmentation score.

    Parameters
    ----------
    query : Spectrum
        The unknown spectrum (must have precursor_mz).
    reference_smiles : str
        SMILES of the best-matching known compound.
    reference_spectrum : Spectrum
        The reference spectrum from the network node (must have precursor_mz).
    mz_tolerance : float
        Fragment matching tolerance in Da.
    delta_mass_tolerance : float
        Tolerance for Δm lookup in Da.
    max_candidates_per_mod : int
        Maximum structural variants explored per modification type.
    top_n : int
        Number of top candidates to return.

    Returns
    -------
    List of StructureCandidate, ranked by fragmentation_score descending.
    Empty list if RDKit is unavailable, Δm is unresolved, or SMILES is invalid.
    """
    if not RDKIT_AVAILABLE:
        warnings.warn("RDKit unavailable — infer_analog_structure returning empty.")
        return []

    query_mz = float(query.get("precursor_mz") or 0.0)
    ref_mz = float(reference_spectrum.get("precursor_mz") or 0.0)
    if not query_mz or not ref_mz:
        warnings.warn("query_mz or ref_mz if empty")
        return []

    delta_mz = query_mz - ref_mz  # signed

    # Step 1: resolve Δm
    modifications = resolve_delta_mass(delta_mz, tolerance_da=delta_mass_tolerance)
    if not modifications:
        warnings.warn("No modification in delta mass")
        return []

    # Step 2: fragment differential (shared across all modifications)
    diff = fragment_differential(query, reference_spectrum, delta_mz, mz_tolerance)

    # Steps 3–5: enumerate → fragment → score
    all_candidates: list[StructureCandidate] = []

    for mod in modifications:
        formula_delta = mod["formula_delta"]
        candidate_smiles_list = enumerate_modified_structures(
            reference_smiles,
            formula_delta,
            max_candidates=max_candidates_per_mod,
        )

        for smi in candidate_smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            predicted = _predict_fragments(mol)
            frag_score = _score_candidate(predicted, query, mz_tolerance)
            all_candidates.append(StructureCandidate(
                smiles=smi,
                formula_delta=formula_delta,
                modification_name=mod["name"],
                fragmentation_score=round(frag_score, 4),
                localization_score=round(diff.localization_score, 4),
                n_conserved_peaks=diff.n_conserved,
                n_shifted_peaks=diff.n_shifted,
            ))

    all_candidates.sort(key=lambda c: -c.fragmentation_score)
    return all_candidates[:top_n]
