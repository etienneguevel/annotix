"""
Unit tests for select_annotation() added to pipeline.py in commit 6c6be16.

select_annotation() decides how to label a spectrum from annotate() results:
  - Scenario 1: direct match, high score, low mass shift → "identified"
  - Scenario 2a: direct match, large mass shift           → "analog"
  - Scenario 2b: propagation, shallow depth, decent score → "putative"
  - Scenario 3: no candidate / low confidence             → None
"""


import pandas as pd

# matchms >= 0.24 renamed ModifiedCosine → ModifiedCosineGreedy.
# Patch the similarity module so pipeline.py can be imported.
import matchms.similarity as _ms_sim
if not hasattr(_ms_sim, "ModifiedCosine"):
    _ms_sim.ModifiedCosine = _ms_sim.ModifiedCosineGreedy

from annotix_ml.propagation.pipeline import select_annotation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(**kwargs) -> pd.DataFrame:
    defaults = {
        "source": "direct_match",
        "score": 0.8,
        "mass_shift_da": 0.005,
        "compound_name": "TestCompound",
        "confidence_level": "MSI-2",
        "propagation_depth": 0,
    }
    defaults.update(kwargs)
    return pd.DataFrame([defaults])


# ===========================================================================
# Tests
# ===========================================================================

class TestSelectAnnotation:

    # --- Scenario 3: empty input ---

    def test_empty_dataframe_returns_none(self):
        assert select_annotation(pd.DataFrame()) is None

    # --- Scenario 1: direct identified match ---

    def test_identified_high_score_low_shift(self):
        result = select_annotation(_row(score=0.9, mass_shift_da=0.001))
        assert result is not None
        assert result["type"] == "identified"
        assert result["annotation"] == "TestCompound"

    def test_identified_score_exactly_threshold(self):
        result = select_annotation(_row(score=0.7, mass_shift_da=0.001))
        assert result["type"] == "identified"

    def test_identified_mass_shift_just_below_threshold(self):
        result = select_annotation(_row(score=0.8, mass_shift_da=0.019))
        assert result["type"] == "identified"

    def test_not_identified_score_below_threshold(self):
        result = select_annotation(_row(score=0.6, mass_shift_da=0.001))
        assert result is None or result["type"] != "identified"

    # --- Scenario 2a: analog (direct match with large mass shift) ---

    def test_analog_large_mass_shift(self):
        result = select_annotation(_row(source="direct_match", mass_shift_da=14.016, score=0.5))
        assert result is not None
        assert result["type"] == "analog"
        assert "TestCompound" in result["annotation"]

    def test_analog_mass_shift_exactly_threshold(self):
        result = select_annotation(_row(source="direct_match", mass_shift_da=0.02, score=0.5))
        assert result is not None
        assert result["type"] == "analog"

    def test_analog_confidence_is_msi3(self):
        result = select_annotation(_row(source="direct_match", mass_shift_da=15.995))
        assert result["confidence"] == "MSI-3"

    # --- Scenario 2b: putative propagation ---

    def test_putative_propagation_shallow(self):
        result = select_annotation(_row(source="propagation", propagation_depth=1, score=0.5))
        assert result is not None
        assert result["type"] == "putative"
        assert "TestCompound" in result["annotation"]

    def test_putative_depth_zero_also_valid(self):
        result = select_annotation(_row(source="propagation", propagation_depth=0, score=0.4))
        assert result["type"] == "putative"

    def test_putative_score_exactly_threshold(self):
        result = select_annotation(_row(source="propagation", propagation_depth=1, score=0.3))
        assert result["type"] == "putative"

    def test_putative_deep_propagation_returns_none(self):
        result = select_annotation(_row(source="propagation", propagation_depth=2, score=0.8))
        assert result is None

    def test_putative_low_score_returns_none(self):
        result = select_annotation(_row(source="propagation", propagation_depth=1, score=0.2))
        assert result is None

    # --- Return structure ---

    def test_result_has_required_keys(self):
        result = select_annotation(_row())
        assert result is not None
        assert "annotation" in result
        assert "confidence" in result
        assert "type" in result
