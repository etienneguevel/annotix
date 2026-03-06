"""
Unit tests for annotix_ml/propagation/analog_inference.py

Covers:
  - resolve_delta_mass
  - fragment_differential / FragmentDiff
  - enumerate_modified_structures  (skipped when RDKit is unavailable)
  - _predict_fragments
  - _score_candidate
  - infer_analog_structure (integration, skipped without RDKit)
"""

import json
from pathlib import Path

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
    from annotix_ml.propagation.analog_inference import (
        _predict_fragments,
        enumerate_modified_structures,
    )
    from rdkit import Chem

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def delta_mass_table():
    table_path = Path(__file__).resolve().parent.parent / "resources" / "delta_mass_table.json"
    with open(table_path) as f:
        return json.load(f)


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

    def test_plus_ch2_exact(self, delta_mass_table):
        hits = resolve_delta_mass(14.0157, delta_mass_table=delta_mass_table)
        assert len(hits) == 1
        assert hits[0]["formula_delta"] == "+CH2"
        assert hits[0]["direction"] == "gain"

    def test_plus_o_exact(self, delta_mass_table):
        hits = resolve_delta_mass(15.9949, delta_mass_table=delta_mass_table)
        assert len(hits) == 1
        assert hits[0]["formula_delta"] == "+O"

    def test_minus_h2o_negative_delta(self, delta_mass_table):
        hits = resolve_delta_mass(-18.0106, delta_mass_table=delta_mass_table)
        assert len(hits) == 1
        assert hits[0]["formula_delta"] == "-H2O"
        assert hits[0]["direction"] == "loss"

    def test_hex_glycosylation(self, delta_mass_table):
        hits = resolve_delta_mass(162.0528, delta_mass_table=delta_mass_table)
        formulas = [h["formula_delta"] for h in hits]
        assert "+Hex" in formulas

    def test_no_match_large_delta(self, delta_mass_table):
        hits = resolve_delta_mass(500.0, delta_mass_table=delta_mass_table)
        assert hits == []

    def test_within_tolerance(self, delta_mass_table):
        hits = resolve_delta_mass(14.0157 + 0.015, delta_mass_table=delta_mass_table)
        assert any(h["formula_delta"] == "+CH2" for h in hits)

    def test_outside_tolerance(self, delta_mass_table):
        hits = resolve_delta_mass(14.0157 + 0.03, delta_mass_table=delta_mass_table)
        assert not any(h["formula_delta"] == "+CH2" for h in hits)

    def test_custom_tight_tolerance(self, delta_mass_table):
        hits = resolve_delta_mass(14.0157 + 0.015, tolerance_da=0.01, delta_mass_table=delta_mass_table)
        assert not any(h["formula_delta"] == "+CH2" for h in hits)

    def test_mass_error_field(self, delta_mass_table):
        hits = resolve_delta_mass(14.0257, delta_mass_table=delta_mass_table)
        ch2_hits = [h for h in hits if h["formula_delta"] == "+CH2"]
        assert len(ch2_hits) == 1
        assert abs(ch2_hits[0]["mass_error_da"]) <= 0.02

    def test_zero_delta(self, delta_mass_table):
        hits = resolve_delta_mass(0.0, delta_mass_table=delta_mass_table)
        assert hits == []

    def test_multiple_matches_at_28da(self, delta_mass_table):
        """28 Da is ambiguous: +CO (27.9949) and +C2H4 (28.0313) differ by 36 mDa."""
        hits_co = resolve_delta_mass(27.9949, delta_mass_table=delta_mass_table)
        assert any(h["formula_delta"] == "+CO" for h in hits_co)
        hits_c2h4 = resolve_delta_mass(28.0313, delta_mass_table=delta_mass_table)
        assert any(h["formula_delta"] == "+C2H4" for h in hits_c2h4)

    def test_direction_gain_for_positive(self, delta_mass_table):
        hits = resolve_delta_mass(15.9949, delta_mass_table=delta_mass_table)
        assert all(h["direction"] == "gain" for h in hits)

    def test_direction_loss_for_negative(self, delta_mass_table):
        hits = resolve_delta_mass(-15.9949, delta_mass_table=delta_mass_table)
        assert all(h["direction"] == "loss" for h in hits)


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
        q_mzs = [100.0, 150.0 + delta, 999.0]
        q = make_spectrum(q_mzs, [1.0] * 3)
        r = make_spectrum(ref_mzs, [1.0] * 3)
        diff = fragment_differential(q, r, delta_mz=delta)
        assert diff.n_conserved == 1
        assert diff.n_shifted == 1
        assert len(diff.novel_mzs) == 1
        assert pytest.approx(diff.localization_score) == 0.5

    def test_lost_peaks(self):
        """Peaks only in reference -> lost_mzs."""
        q = make_spectrum([100.0], [1.0])
        r = make_spectrum([100.0, 300.0, 400.0], [1.0, 1.0, 1.0])
        diff = fragment_differential(q, r, delta_mz=0.0)
        assert len(diff.lost_mzs) == 2
        assert set(diff.lost_mzs) == {300.0, 400.0}

    def test_novel_peaks(self):
        """Peaks only in query (no match in ref, not shifted) -> novel_mzs."""
        q = make_spectrum([100.0, 500.0], [1.0, 1.0])
        r = make_spectrum([100.0], [1.0])
        diff = fragment_differential(q, r, delta_mz=0.0)
        assert len(diff.novel_mzs) == 1
        assert diff.novel_mzs[0] == pytest.approx(500.0)

    def test_localization_score_zero_denominator(self):
        """Empty spectra -> localization_score should be 0.0, not a division error."""
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

    def test_delta_mz_stored(self):
        """The delta_mz value is stored in the result."""
        q = make_spectrum([100.0], [1.0])
        r = make_spectrum([100.0], [1.0])
        diff = fragment_differential(q, r, delta_mz=14.016)
        assert diff.delta_mz == 14.016

    def test_custom_mz_tolerance(self):
        """Wider tolerance should match peaks that default tolerance misses."""
        q = make_spectrum([100.05], [1.0])
        r = make_spectrum([100.0], [1.0])
        diff_tight = fragment_differential(q, r, delta_mz=0.0, mz_tolerance=0.02)
        diff_wide = fragment_differential(q, r, delta_mz=0.0, mz_tolerance=0.1)
        assert diff_tight.n_conserved == 0
        assert diff_wide.n_conserved == 1

    def test_negative_delta(self):
        """Negative delta_mz (unknown is lighter) should still find shifted peaks."""
        ref_mzs = [100.0, 150.0]
        delta = -18.011
        q_mzs = [m + delta for m in ref_mzs]
        q = make_spectrum(q_mzs, [1.0, 1.0])
        r = make_spectrum(ref_mzs, [1.0, 1.0])
        diff = fragment_differential(q, r, delta_mz=delta)
        assert diff.n_shifted == 2


# ===========================================================================
# 3. enumerate_modified_structures  (RDKit required)
# ===========================================================================

@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not installed")
class TestEnumerateModifiedStructures:

    def test_plus_ch2_butane(self):
        results = enumerate_modified_structures("CCCC", "+CH2")
        assert len(results) >= 1
        canonical_pentane = Chem.MolToSmiles(Chem.MolFromSmiles("CCCCC"))
        assert canonical_pentane in results

    def test_plus_o_ethanol(self):
        results = enumerate_modified_structures("CCO", "+O")
        assert len(results) >= 1

    def test_minus_h2o_on_alcohol(self):
        results = enumerate_modified_structures("CC(O)C", "-H2O")
        assert len(results) >= 1

    def test_unsupported_formula_delta(self):
        results = enumerate_modified_structures("CCCC", "+UNKNOWN")
        assert results == []

    def test_invalid_smiles(self):
        results = enumerate_modified_structures("not_a_smiles", "+CH2")
        assert results == []

    def test_max_candidates_respected(self):
        results = enumerate_modified_structures("CCCCCCCCC", "+CH2", max_candidates=3)
        assert len(results) <= 3

    def test_returns_canonical_smiles(self):
        results = enumerate_modified_structures("CCCC", "+CH2")
        for smi in results:
            mol = Chem.MolFromSmiles(smi)
            assert mol is not None, f"Result '{smi}' is not a valid SMILES"

    def test_no_duplicates(self):
        results = enumerate_modified_structures("CCCC", "+CH2")
        assert len(results) == len(set(results))

    def test_plus_acetyl(self):
        results = enumerate_modified_structures("CCO", "+C2H2O")
        assert len(results) >= 1

    def test_plus_co_terminal(self):
        """Formylation on butane terminal carbons."""
        results = enumerate_modified_structures("CCCC", "+CO")
        assert len(results) >= 1

    def test_plus_c2h4(self):
        """Chain elongation by 2 carbons."""
        results = enumerate_modified_structures("CCCC", "+C2H4")
        assert len(results) >= 1

    def test_plus_o2_dihydroxylation(self):
        results = enumerate_modified_structures("CCCC", "+O2")
        assert len(results) >= 1

    def test_plus_o_minus_h2_oxidation(self):
        """Alcohol to ketone: needs an OH group."""
        results = enumerate_modified_structures("CC(O)CC", "+O-H2")
        assert len(results) >= 1

    def test_pm_h2_double_bond_saturation(self):
        """Saturate a C=C bond."""
        results = enumerate_modified_structures("CC=CC", "±H2")
        assert len(results) >= 1
        # At least one result should be fully saturated butane
        canonical_butane = Chem.MolToSmiles(Chem.MolFromSmiles("CCCC"))
        assert canonical_butane in results

    def test_plus_nh3_amination(self):
        results = enumerate_modified_structures("CCCC", "+NH3")
        assert len(results) >= 1

    def test_plus_s_thiolation(self):
        results = enumerate_modified_structures("CCCC", "+S")
        assert len(results) >= 1

    def test_plus_cl_minus_h(self):
        results = enumerate_modified_structures("CCCC", "+Cl-H")
        assert len(results) >= 1

    def test_plus_br_minus_h(self):
        results = enumerate_modified_structures("CCCC", "+Br-H")
        assert len(results) >= 1

    def test_plus_ch2o(self):
        results = enumerate_modified_structures("CCCC", "+CH2O")
        assert len(results) >= 1

    def test_plus_co2_carboxylation(self):
        results = enumerate_modified_structures("CCCC", "+CO2")
        assert len(results) >= 1

    def test_plus_c4h8(self):
        results = enumerate_modified_structures("CCCC", "+C4H8")
        assert len(results) >= 1

    def test_cl_f_swap(self):
        """Replace F with Cl on fluorobenzene."""
        results = enumerate_modified_structures("Fc1ccccc1", "+Cl-F")
        assert len(results) >= 1
        assert any("Cl" in smi for smi in results)

    def test_br_f_swap(self):
        results = enumerate_modified_structures("Fc1ccccc1", "+Br-F")
        assert len(results) >= 1
        assert any("Br" in smi for smi in results)

    def test_double_cl_f_swap(self):
        """Molecule with 2 F atoms -> double Cl/F swap."""
        results = enumerate_modified_structures("Fc1ccc(F)cc1", "+2Cl-2F")
        assert len(results) >= 1

    def test_double_cl_h(self):
        results = enumerate_modified_structures("CCCC", "+2Cl-2H")
        assert len(results) >= 1

    def test_no_modification_on_aromatic_for_hydroxylation(self):
        """Aromatic carbons should be skipped by +O hydroxylation."""
        results = enumerate_modified_structures("c1ccccc1", "+O")
        assert results == []

    def test_empty_molecule_ch2(self):
        """Methane has no terminal heavy-atom-degree-1 carbon with heavy neighbors."""
        results = enumerate_modified_structures("C", "+CH2")
        # Methane's single carbon has no heavy neighbors, so no extension
        assert results == []


# ===========================================================================
# 4. _predict_fragments  (RDKit required)
# ===========================================================================

@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not installed")
class TestPredictFragments:

    def test_returns_sorted_list(self):
        mol = Chem.MolFromSmiles("CCCCCC")
        frags = _predict_fragments(mol)
        assert frags == sorted(frags)

    def test_returns_unique_values(self):
        mol = Chem.MolFromSmiles("CCCCCC")
        frags = _predict_fragments(mol)
        assert len(frags) == len(set(frags))

    def test_includes_neutral_losses(self):
        """Should include at least some neutral-loss peaks from the precursor."""
        mol = Chem.MolFromSmiles("CCCCCCC")
        frags = _predict_fragments(mol)
        assert len(frags) > 0

    def test_max_fragments_cap(self):
        """With a low cap, fewer fragments are produced than with default."""
        mol = Chem.MolFromSmiles("C" * 20)
        frags_capped = _predict_fragments(mol, max_fragments=5)
        frags_default = _predict_fragments(mol, max_fragments=150)
        assert len(frags_capped) <= len(frags_default)

    def test_ring_bonds_not_cleaved(self):
        """Cyclohexane has only ring bonds -> only neutral losses should appear."""
        mol = Chem.MolFromSmiles("C1CCCCC1")
        frags = _predict_fragments(mol)
        # Only neutral losses (4 items), no bond-cleavage fragments
        assert len(frags) == 4

    def test_ethane_two_fragments(self):
        """Ethane: one acyclic single bond -> two fragment ions + neutral losses."""
        mol = Chem.MolFromSmiles("CC")
        frags = _predict_fragments(mol)
        assert len(frags) >= 2


# ===========================================================================
# 5. _score_candidate
# ===========================================================================

class TestScoreCandidate:

    def test_perfect_match(self):
        """All predicted peaks present in query -> score close to 1.0."""
        mzs = [100.0, 150.0, 200.0]
        q = make_spectrum(mzs, [1.0, 1.0, 1.0])
        score = _score_candidate(mzs, q)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_no_match(self):
        """No predicted peak overlaps query -> score = 0.0."""
        q = make_spectrum([100.0, 150.0], [1.0, 1.0])
        score = _score_candidate([500.0, 600.0], q)
        assert score == pytest.approx(0.0)

    def test_empty_predicted(self):
        q = make_spectrum([100.0], [1.0])
        assert _score_candidate([], q) == 0.0

    def test_partial_match(self):
        """Half the intensity matched -> score ~ 0.5."""
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
        score_in = _score_candidate([100.02], q, mz_tolerance=0.02)
        assert score_in == pytest.approx(1.0)
        score_out = _score_candidate([100.03], q, mz_tolerance=0.02)
        assert score_out == pytest.approx(0.0)

    def test_intensity_weighting(self):
        """Matching the high-intensity peak should give a higher score than matching the low one."""
        q = make_spectrum([100.0, 200.0], [10.0, 1.0])
        score_high = _score_candidate([100.0], q)
        score_low = _score_candidate([200.0], q)
        assert score_high > score_low


# ===========================================================================
# 6. infer_analog_structure (integration)
# ===========================================================================

@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not installed")
class TestInferAnalogStructure:

    @pytest.fixture
    def delta_mass_table(self):
        table_path = Path(__file__).resolve().parent.parent / "resources" / "delta_mass_table.json"
        with open(table_path) as f:
            return json.load(f)

    def _make_analog_pair(self, ref_smiles: str, delta_da: float, n_peaks: int = 5):
        ref_mzs = np.linspace(50, 150, n_peaks)
        ref_ints = np.ones(n_peaks)
        ref_prec = 200.0
        q_prec = ref_prec + delta_da

        q_mzs = ref_mzs + delta_da
        q_ints = np.ones(n_peaks)

        ref_spec = make_spectrum(ref_mzs.tolist(), ref_ints.tolist(), precursor_mz=ref_prec)
        q_spec = make_spectrum(q_mzs.tolist(), q_ints.tolist(), precursor_mz=q_prec)
        return q_spec, ref_spec

    def test_returns_list(self, delta_mass_table):
        q, r = self._make_analog_pair("CCCC", 14.0157)
        result = infer_analog_structure(q, "CCCC", r, delta_mass_table=delta_mass_table)
        assert isinstance(result, list)

    def test_top_n_respected(self, delta_mass_table):
        q, r = self._make_analog_pair("CCCC", 14.0157)
        result = infer_analog_structure(q, "CCCC", r, top_n=2, delta_mass_table=delta_mass_table)
        assert len(result) <= 2

    def test_candidates_are_structure_candidates(self, delta_mass_table):
        q, r = self._make_analog_pair("CCCC", 14.0157)
        result = infer_analog_structure(q, "CCCC", r, delta_mass_table=delta_mass_table)
        for c in result:
            assert isinstance(c, StructureCandidate)

    def test_sorted_by_fragmentation_score(self, delta_mass_table):
        q, r = self._make_analog_pair("CCCCC", 14.0157)
        result = infer_analog_structure(q, "CCCCC", r, top_n=5, delta_mass_table=delta_mass_table)
        scores = [c.fragmentation_score for c in result]
        assert scores == sorted(scores, reverse=True)

    def test_empty_when_delta_not_resolved(self, delta_mass_table):
        """Precursor delta of 1000 Da -> no modification found -> empty."""
        q, r = self._make_analog_pair("CCCC", 1000.0)
        result = infer_analog_structure(q, "CCCC", r, delta_mass_table=delta_mass_table)
        assert result == []

    def test_empty_when_missing_precursor_mz(self, delta_mass_table):
        q = Spectrum(
            mz=np.array([100.0]),
            intensities=np.array([1.0]),
            metadata={},
        )
        r = make_spectrum([100.0], [1.0], precursor_mz=200.0)
        result = infer_analog_structure(q, "CCCC", r, delta_mass_table=delta_mass_table)
        assert result == []

    def test_empty_when_invalid_smiles(self, delta_mass_table):
        q, r = self._make_analog_pair("CCCC", 14.0157)
        result = infer_analog_structure(q, "NOT_VALID_SMILES", r, delta_mass_table=delta_mass_table)
        assert result == []

    def test_msi_level_field(self, delta_mass_table):
        q, r = self._make_analog_pair("CCCC", 14.0157)
        result = infer_analog_structure(q, "CCCC", r, delta_mass_table=delta_mass_table)
        for c in result:
            assert "MSI" in c.msi_level

    def test_formula_delta_populated(self, delta_mass_table):
        """Each candidate should carry the applied formula delta."""
        q, r = self._make_analog_pair("CCCC", 14.0157)
        result = infer_analog_structure(q, "CCCC", r, delta_mass_table=delta_mass_table)
        for c in result:
            assert c.formula_delta == "+CH2"

    def test_modification_name_populated(self, delta_mass_table):
        q, r = self._make_analog_pair("CCCC", 14.0157)
        result = infer_analog_structure(q, "CCCC", r, delta_mass_table=delta_mass_table)
        for c in result:
            assert len(c.modification_name) > 0

    def test_fragmentation_score_in_range(self, delta_mass_table):
        q, r = self._make_analog_pair("CCCC", 14.0157)
        result = infer_analog_structure(q, "CCCC", r, delta_mass_table=delta_mass_table)
        for c in result:
            assert 0.0 <= c.fragmentation_score <= 1.0

    def test_plus_o_hydroxylation(self, delta_mass_table):
        """Integration test with +O (hydroxylation) on a chain molecule."""
        q, r = self._make_analog_pair("CCCCCC", 15.9949)
        result = infer_analog_structure(q, "CCCCCC", r, delta_mass_table=delta_mass_table)
        assert isinstance(result, list)
        if result:
            assert all(c.formula_delta == "+O" for c in result)

    def test_conserved_and_shifted_peaks_reported(self, delta_mass_table):
        """n_conserved_peaks and n_shifted_peaks should be non-negative integers."""
        q, r = self._make_analog_pair("CCCC", 14.0157)
        result = infer_analog_structure(q, "CCCC", r, delta_mass_table=delta_mass_table)
        for c in result:
            assert isinstance(c.n_conserved_peaks, int)
            assert isinstance(c.n_shifted_peaks, int)
            assert c.n_conserved_peaks >= 0
            assert c.n_shifted_peaks >= 0

    def test_localization_score_in_range(self, delta_mass_table):
        q, r = self._make_analog_pair("CCCC", 14.0157)
        result = infer_analog_structure(q, "CCCC", r, delta_mass_table=delta_mass_table)
        for c in result:
            assert 0.0 <= c.localization_score <= 1.0
