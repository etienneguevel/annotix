# Analog Structure Inference — Technical Documentation

This document describes the `analog_inference.py` module, which infers the molecular structure of an unknown compound from its MS/MS spectrum when the compound is a structural analog of a known reference.

---

## Problem Statement

When a molecular networking pipeline identifies an unknown spectrum as a **structural analog** of a known compound (same fragmentation pattern, different precursor mass), the question becomes: *what is the actual structure of the unknown?*

The precursor mass difference (delta-m) between the unknown and its closest known neighbor encodes a molecular formula change. This module exploits that information to:

1. Identify *what* chemical modification occurred (e.g. +CH2, +O, -H2O).
2. Enumerate *where* on the reference scaffold the modification could be placed.
3. Rank candidates by how well their predicted fragments match the observed spectrum.

---

## Pipeline Overview

```
Unknown spectrum + Reference SMILES + Reference spectrum
                        |
         Step 1: Compute signed delta-m
                        |
         Step 2: Resolve delta-m to biochemical modification(s)
                        |
         Step 3: Fragment differential analysis
                        |
         Step 4: Enumerate modified structures (RDKit)
                        |
         Step 5: In silico fragmentation (bond cleavage)
                        |
         Step 6: Score predicted vs. observed fragments
                        |
              Ranked StructureCandidate list
```

---

## Functions Reference

### `resolve_delta_mass(delta_da, tolerance_da, delta_mass_table)`

**Purpose:** Map the signed precursor mass shift to known biochemical modifications using a lookup table.

**How it works:**
- Takes the absolute value of the mass shift and compares it against each entry in the delta mass table.
- A match occurs when `|abs(delta_da) - exact_mass| <= tolerance_da`.
- The sign of `delta_da` determines the `direction` field: positive means the unknown is heavier ("gain"), negative means lighter ("loss").
- Returns all matching entries (there can be multiple matches at unit mass resolution, e.g. +CO and +C2H4 both near 28 Da).

**Parameters:**
| Name | Type | Default | Description |
|---|---|---|---|
| `delta_da` | float | — | Signed mass shift: query_precursor_mz - reference_precursor_mz |
| `tolerance_da` | float | 0.02 | Matching tolerance in Daltons |
| `delta_mass_table` | list[tuple] | None | Table of (exact_mass, formula_tag, name) tuples, loaded from `resources/delta_mass_table.json` |

**Returns:** List of dicts with keys: `exact_mass_da`, `formula_delta`, `name`, `direction`, `mass_error_da`.

---

### `fragment_differential(query, reference, delta_mz, mz_tolerance)`

**Purpose:** Classify each fragment peak of the unknown spectrum relative to the reference spectrum.

**How it works:**

For each peak in the query spectrum (iterating over m/z values):

1. **Conserved check:** Is there a reference peak within `mz_tolerance` of the same m/z? If yes, the peak is **conserved** — it belongs to the unmodified part of the scaffold.

2. **Shifted check:** Is there a reference peak within `mz_tolerance` of `(query_mz - delta_mz)`? If yes, the peak is **shifted** — it contains the modification site. The logic: if a reference fragment at mass `r` gains the modification delta, the query fragment appears at `r + delta`, so `r = query_mz - delta`.

3. **Novel:** If neither check matches, the peak is **novel** — it arises from fragmentation of the modified region and has no counterpart in the reference.

4. **Lost:** Reference peaks that were never matched by any query peak (neither as conserved nor as shifted) are **lost** — the modification eliminated this fragmentation pathway.

**Key output — `localization_score`:**

```
localization_score = n_shifted / (n_conserved + n_shifted)
```

- **0.0** = all matched peaks are conserved; the modification does not appear in the fragment ion series (e.g. a terminal chain extension).
- **1.0** = all matched peaks are shifted; every fragment carries the modification (e.g. modification is in the core scaffold).
- Values in between indicate partial localization.

**Returns:** A `FragmentDiff` dataclass containing `conserved_mzs`, `shifted_mzs`, `novel_mzs`, `lost_mzs`, `delta_mz`, `n_conserved`, `n_shifted`, and `localization_score`.

---

### `enumerate_modified_structures(smiles, formula_delta, max_candidates)`

**Purpose:** Generate plausible structural modifications of a known compound by applying the identified formula delta at every chemically valid position.

**How it works:**

1. Parse the reference SMILES into an RDKit molecule object.
2. Look up the `formula_delta` tag in the `_ENUMERATION_DISPATCH` table to find the appropriate enumeration function.
3. The enumeration function applies the modification at each eligible site on the molecule, producing a list of candidate molecules.
4. Candidates are deduplicated by canonical SMILES and capped at `max_candidates`.

**Supported modifications and their enumeration strategies:**

| Formula delta | Exact mass (Da) | Strategy |
|---|---|---|
| `+CH2` | 14.016 | Extend each terminal carbon by one CH2 |
| `+O` | 15.995 | Add OH to each aliphatic CH carbon (hydroxylation) |
| `-H2O` | 18.011 | Beta-elimination via SMARTS: `[C][C][OH] >> [C]=[C]` |
| `+CO` | 27.995 | Append CHO (formyl) to each terminal carbon |
| `+C2H4` | 28.031 | Two successive +CH2 extensions |
| `+C2H2O` | 42.011 | Acetylation of OH or NH groups via SMARTS reactions |
| `+O2` | 31.990 | Double hydroxylation (two successive +O) |
| `+O-H2` | 13.979 | Alcohol-to-ketone oxidation via SMARTS: `[C][OH] >> [C]=O` |
| `+/-H2` | 2.016 | Saturate one non-aromatic C=C double bond |
| `+NH3` | 17.027 | Add NH2 to each aliphatic CH carbon |
| `+S` | 31.972 | Add SH to each aliphatic CH carbon |
| `+Cl-H` | 33.961 | Replace one H on an aliphatic C with Cl |
| `+Br-H` | 77.911 | Replace one H on an aliphatic C with Br |
| `+CH2O` | 30.011 | Add CH2OH to each terminal carbon |
| `+CO2` | 43.990 | Add COOH to each terminal carbon |
| `+C4H8` | 56.063 | Four successive +CH2 (lipid chain elongation) |
| `+Cl-F` | 15.971 | Replace one F with Cl (halogen swap) |
| `+Br-F` | 43.950 | Replace one F with Br (halogen swap) |
| `+2Cl-2F` | 21.982 | Two Cl/F swaps |
| `+2Cl-2H` | 51.941 | Two chlorinations |

**Two enumeration approaches are used:**

- **Atom-level manipulation** (most modifications): iterate over atoms, check eligibility (element, aromaticity, hydrogen count, degree), then add/modify atoms and bonds using `RWMol`.
- **SMARTS reactions** (`-H2O`, `+C2H2O`, `+O-H2`): use `AllChem.ReactionFromSmarts` for transformations that involve bond rearrangement (e.g. elimination, oxidation).

**Returns:** List of canonical SMILES strings (empty if modification unsupported or SMILES invalid).

---

### `_predict_fragments(mol, max_fragments)`

**Purpose:** Predict fragment m/z values for a candidate molecule using simplified bond cleavage.

**How it works:**

1. Identify all **acyclic single bonds** (ring bonds are skipped — ring opening requires multi-step mechanisms not modeled here).
2. For each eligible bond, remove it to split the molecule into two fragments.
3. Record each fragment as a protonated ion: `m/z = exact_mass + 1.00728` (the proton mass for [M+H]+ ions).
4. Add common **neutral losses** from the precursor [M+H]+: -H2O (18.011), -NH3 (17.027), -CO (27.995), -CO2 (44.026).

**Limitations:** This is a first-order approximation. It does not model rearrangements, ring openings, charge-directed fragmentation, or multi-step mechanisms. For production use, CFM-ID or MetFrag would provide more accurate predictions.

**Returns:** Sorted list of unique predicted m/z values (rounded to 4 decimal places).

---

### `_score_candidate(predicted_mzs, query, mz_tolerance)`

**Purpose:** Score how well a candidate's predicted fragments explain the observed spectrum.

**How it works:**

1. Normalize the query spectrum intensities to sum to 1.
2. For each query peak, check if any predicted m/z falls within `mz_tolerance`.
3. Sum the normalized intensities of all matched peaks.

```
score = sum(intensity of matched peaks) / sum(all intensities)
```

This is an **intensity-weighted recall**: it measures what fraction of the observed signal is explained by the candidate's predicted fragments. Higher-intensity peaks contribute more to the score.

**Returns:** Float in [0, 1]. A score of 1.0 means every observed peak has a matching prediction.

---

### `infer_analog_structure(query, reference_smiles, reference_spectrum, ...)`

**Purpose:** Main entry point. Orchestrates the full pipeline from spectra to ranked structure candidates.

**How it works:**

1. Extract precursor m/z from both spectra and compute `delta_mz = query_mz - ref_mz`.
2. Call `resolve_delta_mass` to find matching biochemical modifications.
3. Call `fragment_differential` to classify peaks (shared across all modifications).
4. For each resolved modification:
   - Enumerate candidate structures with `enumerate_modified_structures`.
   - For each candidate, predict fragments with `_predict_fragments`.
   - Score against the query with `_score_candidate`.
5. Sort all candidates by `fragmentation_score` descending and return the top N.

**Parameters:**
| Name | Type | Default | Description |
|---|---|---|---|
| `query` | Spectrum | — | Unknown spectrum (must have `precursor_mz`) |
| `reference_smiles` | str | — | SMILES of the known reference compound |
| `reference_spectrum` | Spectrum | — | Reference spectrum (must have `precursor_mz`) |
| `mz_tolerance` | float | 0.02 | Fragment matching tolerance (Da) |
| `delta_mass_tolerance` | float | 0.02 | Tolerance for delta-m lookup (Da) |
| `max_candidates_per_mod` | int | 10 | Max structural variants per modification type |
| `top_n` | int | 5 | Number of top candidates to return |
| `delta_mass_table` | list | None | Delta mass lookup table |

**Returns:** List of `StructureCandidate` dataclass instances, each containing:
- `smiles` — canonical SMILES of the candidate
- `formula_delta` — the applied modification tag
- `modification_name` — human-readable name
- `fragmentation_score` — fraction of observed intensity explained [0-1]
- `localization_score` — fraction of matched peaks that are shifted [0-1]
- `n_conserved_peaks`, `n_shifted_peaks`
- `msi_level` — defaults to "MSI-3 (putative analog)"

---

## Data Flow Diagram

```
                    query_spectrum.precursor_mz
                    reference_spectrum.precursor_mz
                              |
                    delta_mz = query - ref
                              |
                   resolve_delta_mass(delta_mz)
                      /               \
            ["+CH2", "+O", ...]   fragment_differential(query, ref, delta_mz)
                |                          |
                |                   FragmentDiff (localization_score)
                |                          |
     enumerate_modified_structures    (shared by all candidates)
          for each modification
                |
         [SMILES_1, SMILES_2, ...]
                |
         _predict_fragments(mol)
                |
         _score_candidate(predicted, query)
                |
         StructureCandidate(smiles, score, ...)
                |
         sort by fragmentation_score → top_n
```

---

## Integration with the Annotation Pipeline

This module is called from `inference.py` via the `infer_structure` function:

1. `find_candidates()` runs direct matching + annotation propagation, returning a DataFrame of ranked candidates.
2. For each candidate with a non-zero mass shift (i.e. not an exact match), `infer_analog_structure` is called to propose modified structures.
3. The script `main_inference.py` orchestrates this: if no exact match is found, it calls `infer_structure` which loops over all candidates.

---

## Delta Mass Table

The lookup table (`resources/delta_mass_table.json`) contains 37 entries covering:
- **Oxidation/reduction:** +O, +O2, +O-H2, +/-H2
- **Carbon extensions:** +CH2, +C2H4, +C4H8, +CO, +CO2, +CH2O
- **Heteroatom additions:** +NH3, +S, +Cl-H, +Br-H
- **Conjugations:** +C2H2O (acetylation), +Gly, +GSH
- **Glycosylation:** +Pent, +dHex, +Hex, +GlcA, +HexNAc
- **Halogen swaps:** +Cl-F, +Br-F, +2Cl-2F, +2Cl-2H
- **Dehydration:** -H2O

Not all entries have corresponding enumeration functions in `_ENUMERATION_DISPATCH` (20 of 37 are implemented). Unmatched formula deltas return empty candidate lists gracefully.
