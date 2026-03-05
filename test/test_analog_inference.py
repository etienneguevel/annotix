"""
Unit tests for annotix_ml/propagation/analog_inference.py

Covers:
  - resolve_delta_mass
  - fragment_differential / FragmentDiff
  - enumerate_modified_structures  (skipped when RDKit is unavailable)
  - _score_candidate
  - infer_analog_structure (integration, skipped without RDKit)
"""

import numpy as np
import pytest
from matchms import Spectrum

from annotix_ml.propagation.analog_inference import (
    RDKIT_AVAILABLE,
    FragmentDiff,
    StructureCandidate,
    _score_candidate,
    fragment_differential,
    infer_analog_structure,
    resolve_delta_mass,
)

if RDKIT_AVAILABLE:
    from annotix_ml.propagation.analog_inference import enumerate_modified_structures

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_spectrum(mzs: list[float], intensities: list[float], precursor_mz: float = 200.0) -> Spectrum:
    return Spectrum(
        mz=np.array(mzs, dtype=np.float64),
        intensities=np.array(intensities, dtype=np.float64),
        metadata={"precursor_mz": precursor_mz},
    )


# ===========================================================================
# 1. resolve_delta_mass
# ===========================================================================

class TestResolveDeltaMass:

    def test_plus_ch2_exact(self):
        hits = resolve_delta_mass(14.0157)
        assert len(hits) == 1
        assert hits[0]["formula_delta"] == "+CH2"
        assert hits[0]["direction"] == "gain"

    def test_plus_o_exact(self):
        hits = resolve_delta_mass(15.9949)
        assert len(hits) == 1
        assert hits[0]["formula_delta"] == "+O"

    def test_minus_h2o_negative_delta(self):
        # -H2O shows as a loss: delta_da < 0, abs matches 18.0106
        hits = resolve_delta_mass(-18.0106)
        assert len(hits) == 1
        assert hits[0]["formula_delta"] == "-H2O"
        assert hits[0]["direction"] == "loss"

    def test_hex_glycosylation(self):
        hits = resolve_delta_mass(162.0528)
        formulas = [h["formula_delta"] for h in hits]
        assert "+Hex" in formulas

    def test_no_match_large_delta(self):
        hits = resolve_delta_mass(500.0)
        assert hits == []

    def test_within_tolerance(self):
        # Shift +CH2 by 0.015 Da (within default 0.02 tolerance)
        hits = resolve_delta_mass(14.0157 + 0.015)
        assert any(h["formula_delta"] == "+CH2" for h in hits)

    def test_outside_tolerance(self):
        hits = resolve_delta_mass(14.0157 + 0.03)
        assert not any(h["formula_delta"] == "+CH2" for h in hits)

    def test_custom_tight_tolerance(self):
        hits = resolve_delta_mass(14.0157 + 0.015, tolerance_da=0.01)
        assert not any(h["formula_delta"] == "+CH2" for h in hits)

    def test_mass_error_field(self):
        hits = resolve_delta_mass(14.0257)  # 0.01 Da off
        ch2_hits = [h for h in hits if h["formula_delta"] == "+CH2"]
        assert len(ch2_hits) == 1
        assert abs(ch2_hits[0]["mass_error_da"]) <= 0.02

    def test_zero_delta(self):
        hits = resolve_delta_mass(0.0)
        assert hits == []


# ===========================================================================
# 2. fragment_differential
# ===========================================================================

class TestFragmentDifferential:

    def test_all_conserved(self):
        """When query == reference, every peak should be conserved."""
        mzs = [100.0, 150.0, 200.0]
        ints = [1.0, 1.0, 1.0]
        q = make_spectrum(mzs, ints)
        r = make_spectrum(mzs, ints)
        diff = fragment_differential(q, r, delta_mz=0.0)
        assert diff.n_conserved == 3
        assert diff.n_shifted == 0
        assert len(diff.novel_mzs) == 0
        assert diff.localization_score == 0.0

    def test_all_shifted(self):
        """Every query peak is shifted by delta relative to reference."""
        ref_mzs = [100.0, 150.0, 200.0]
        delta = 14.0157
        q_mzs = [m + delta for m in ref_mzs]
        q = make_spectrum(q_mzs, [1.0] * 3)
        r = make_spectrum(ref_mzs, [1.0] * 3)
        diff = fragment_differential(q, r, delta_mz=delta)
        assert diff.n_shifted == 3
        assert diff.n_conserved == 0
        assert diff.localization_score == 1.0

    def test_mixed_conserved_and_shifted(self):
        ref_mzs = [100.0, 150.0, 200.0]
        delta = 14.0157
        # First peak conserved, second shifted, third novel
        q_mzs = [100.0, 150.0 + delta, 999.0]
        q = make_spectrum(q_mzs, [1.0] * 3)
        r = make_spectrum(ref_mzs, [1.0] * 3)
        diff = fragment_differential(q, r, delta_mz=delta)
        assert diff.n_conserved == 1
        assert diff.n_shifted == 1
        assert len(diff.novel_mzs) == 1
        assert pytest.approx(diff.localization_score) == 0.5

    def test_lost_peaks(self):
        """Peaks only in reference → lost_mzs."""
        q = make_spectrum([100.0], [1.0])
        r = make_spectrum([100.0, 300.0, 400.0], [1.0, 1.0, 1.0])
        diff = fragment_differential(q, r, delta_mz=0.0)
        assert len(diff.lost_mzs) == 2
        assert set(diff.lost_mzs) == {300.0, 400.0}

    def test_novel_peaks(self):
        """Peaks only in query (no match in ref, not shifted) → novel_mzs."""
        q = make_spectrum([100.0, 500.0], [1.0, 1.0])
        r = make_spectrum([100.0], [1.0])
        diff = fragment_differential(q, r, delta_mz=0.0)
        assert len(diff.novel_mzs) == 1
        assert diff.novel_mzs[0] == pytest.approx(500.0)

    def test_localization_score_zero_denominator(self):
        """Empty spectra → localization_score should be 0.0, not a division error."""
        q = make_spectrum([], [])
        r = make_spectrum([], [])
        diff = fragment_differential(q, r, delta_mz=14.0)
        assert diff.localization_score == 0.0

    def test_dataclass_fields(self):
        q = make_spectrum([100.0], [1.0])
        r = make_spectrum([100.0], [1.0])
        diff = fragment_differential(q, r, delta_mz=0.0)
        assert isinstance(diff, FragmentDiff)
        assert hasattr(diff, "conserved_mzs")
        assert hasattr(diff, "shifted_mzs")
        assert hasattr(diff, "novel_mzs")
        assert hasattr(diff, "lost_mzs")
        assert hasattr(diff, "localization_score")


# ===========================================================================
# 3. enumerate_modified_structures  (RDKit required)
# ===========================================================================

@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not installed")
class TestEnumerateModifiedStructures:

    def test_plus_ch2_butane(self):
        results = enumerate_modified_structures("CCCC", "+CH2")
        assert len(results) >= 1
        # Pentane canonical SMILES must be among the results
        from rdkit import Chem
        canonical_pentane = Chem.MolToSmiles(Chem.MolFromSmiles("CCCCC"))
        assert canonical_pentane in results

    def test_plus_o_ethanol(self):
        results = enumerate_modified_structures("CCO", "+O")
        assert len(results) >= 1

    def test_minus_h2o_on_alcohol(self):
        # Propanol: CC(O)C → elimination should produce propene CC=C or C=CC
        results = enumerate_modified_structures("CC(O)C", "-H2O")
        assert len(results) >= 1

    def test_unsupported_formula_delta(self):
        results = enumerate_modified_structures("CCCC", "+UNKNOWN")
        assert results == []

    def test_invalid_smiles(self):
        results = enumerate_modified_structures("not_a_smiles", "+CH2")
        assert results == []

    def test_max_candidates_respected(self):
        # Use a large molecule likely to produce many +CH2 candidates
        results = enumerate_modified_structures("CCCCCCCCC", "+CH2", max_candidates=3)
        assert len(results) <= 3

    def test_returns_canonical_smiles(self):
        from rdkit import Chem
        results = enumerate_modified_structures("CCCC", "+CH2")
        for smi in results:
            mol = Chem.MolFromSmiles(smi)
            assert mol is not None, f"Result '{smi}' is not a valid SMILES"

    def test_no_duplicates(self):
        results = enumerate_modified_structures("CCCC", "+CH2")
        assert len(results) == len(set(results))

    def test_plus_acetyl(self):
        # Ethanol has an OH → acetylation should work
        results = enumerate_modified_structures("CCO", "+C2H2O")
        assert len(results) >= 1


# ===========================================================================
# 4. _score_candidate
# ===========================================================================

class TestScoreCandidate:

    def test_perfect_match(self):
        """All predicted peaks present in query → score close to 1.0."""
        mzs = [100.0, 150.0, 200.0]
        q = make_spectrum(mzs, [1.0, 1.0, 1.0])
        score = _score_candidate(mzs, q)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_no_match(self):
        """No predicted peak overlaps query → score = 0.0."""
        q = make_spectrum([100.0, 150.0], [1.0, 1.0])
        score = _score_candidate([500.0, 600.0], q)
        assert score == pytest.approx(0.0)

    def test_empty_predicted(self):
        q = make_spectrum([100.0], [1.0])
        assert _score_candidate([], q) == 0.0

    def test_partial_match(self):
        """Half the intensity matched → score ≈ 0.5."""
        q = make_spectrum([100.0, 200.0], [1.0, 1.0])
        score = _score_candidate([100.0, 999.0], q)
        assert 0.4 < score < 0.6

    def test_score_bounded_zero_one(self):
        q = make_spectrum([100.0, 200.0, 300.0], [3.0, 1.0, 1.0])
        score = _score_candidate([100.0, 200.0, 300.0, 400.0], q)
        assert 0.0 <= score <= 1.0

    def test_zero_intensity_spectrum(self):
        q = make_spectrum([100.0], [0.0])
        assert _score_candidate([100.0], q) == 0.0

    def test_tolerance_boundary(self):
        """Peak exactly at tolerance edge should be matched."""
        q = make_spectrum([100.0], [1.0])
        # Default mz_tolerance=0.02: 100.02 should still match
        score_in = _score_candidate([100.02], q, mz_tolerance=0.02)
        assert score_in == pytest.approx(1.0)
        # 100.03 is outside → no match
        score_out = _score_candidate([100.03], q, mz_tolerance=0.02)
        assert score_out == pytest.approx(0.0)


# ===========================================================================
# 5. infer_analog_structure (integration)
# ===========================================================================

@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not installed")
class TestInferAnalogStructure:

    def _make_analog_pair(self, ref_smiles: str, delta_da: float, n_peaks: int = 5):
        """
        Build a minimal (reference_smiles, reference_spectrum, query_spectrum) triple
        where the query is offset by delta_da on the precursor.
        """
        ref_mzs = np.linspace(50, 150, n_peaks)
        ref_ints = np.ones(n_peaks)
        ref_prec = 200.0
        q_prec = ref_prec + delta_da

        # Shift all fragment peaks by delta too (simulating a fully-shifted spectrum)
        q_mzs = ref_mzs + delta_da
        q_ints = np.ones(n_peaks)

        ref_spec = make_spectrum(ref_mzs.tolist(), ref_ints.tolist(), precursor_mz=ref_prec)
        q_spec = make_spectrum(q_mzs.tolist(), q_ints.tolist(), precursor_mz=q_prec)
        return q_spec, ref_spec

    def test_returns_list(self):
        q, r = self._make_analog_pair("CCCC", 14.0157)
        result = infer_analog_structure(q, "CCCC", r)
        assert isinstance(result, list)

    def test_top_n_respected(self):
        q, r = self._make_analog_pair("CCCC", 14.0157)
        result = infer_analog_structure(q, "CCCC", r, top_n=2)
        assert len(result) <= 2

    def test_candidates_are_structure_candidates(self):
        q, r = self._make_analog_pair("CCCC", 14.0157)
        result = infer_analog_structure(q, "CCCC", r)
        for c in result:
            assert isinstance(c, StructureCandidate)

    def test_sorted_by_fragmentation_score(self):
        q, r = self._make_analog_pair("CCCCC", 14.0157)
        result = infer_analog_structure(q, "CCCCC", r, top_n=5)
        scores = [c.fragmentation_score for c in result]
        assert scores == sorted(scores, reverse=True)

    def test_empty_when_delta_not_resolved(self):
        """Precursor delta of 1000 Da → no modification found → empty."""
        q, r = self._make_analog_pair("CCCC", 1000.0)
        result = infer_analog_structure(q, "CCCC", r)
        assert result == []

    def test_empty_when_missing_precursor_mz(self):
        q = Spectrum(
            mz=np.array([100.0]),
            intensities=np.array([1.0]),
            metadata={},  # no precursor_mz
        )
        r = make_spectrum([100.0], [1.0], precursor_mz=200.0)
        result = infer_analog_structure(q, "CCCC", r)
        assert result == []

    def test_empty_when_invalid_smiles(self):
        q, r = self._make_analog_pair("CCCC", 14.0157)
        result = infer_analog_structure(q, "NOT_VALID_SMILES", r)
        assert result == []

    def test_msi_level_field(self):
        q, r = self._make_analog_pair("CCCC", 14.0157)
        result = infer_analog_structure(q, "CCCC", r)
        for c in result:
            assert "MSI" in c.msi_level
