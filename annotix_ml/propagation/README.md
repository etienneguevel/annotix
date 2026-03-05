# Spectral Annotation Pipeline

MS/MS spectral annotation pipeline using molecular networking and annotation propagation.

---

## Overview

The pipeline annotates unknown MS/MS spectra in two steps:

1. **Direct spectral matching** — compare the unknown against all spectra in a reference molecular network using modified cosine similarity.
2. **Annotation propagation** — if no direct match is found, walk the network from the closest anchor nodes and propagate annotations from neighboring known compounds.

The output is a ranked list of candidate annotations with confidence scores, provenance, and MSI confidence levels.

---

## Methodology

### 1. Molecular Network Construction

The reference molecular network is a graph where:
- **Nodes** are MS/MS spectra (with optional library annotations: compound name, SMILES, InChI, formula, adduct).
- **Edges** connect spectra pairs whose modified cosine similarity exceeds a threshold.

Construction follows the **GNPS/FBMN standard** (Wang et al., *Nat. Biotechnol.* 2016; Nothias et al., *Nat. Methods* 2020):

| Parameter | Default | Description |
|---|---|---|
| `cosine_threshold` | 0.7 | Minimum modified cosine score for an edge |
| `min_matched_peaks` | 6 | Minimum shared fragment peaks |
| `top_k` | 10 | Maximum neighbors per node (mutual top-K filter) |
| `mz_tolerance` | 0.02 Da | Fragment matching tolerance |
| `max_mz_diff` | 500 Da | Pre-filter: skip pairs with large precursor gap |

**Modified Cosine similarity** accounts for a precursor mass offset between the two spectra, so spectra from structural analogs (same scaffold, different mass) can still be matched. This is what allows analog edges to form in the network.

**Top-K filtering** (mutual): an edge between nodes A and B is kept only if B appears in A's top-10 neighbors AND A appears in B's top-10 neighbors. This prevents "hairball" networks of weakly connected clusters.

Pairwise similarity is computed in parallel using shared memory (POSIX fork on macOS/Linux, spawn on Windows) and per-node min-heaps for O(1) top-K maintenance.

### 2. Direct Spectral Matching

The unknown spectrum is compared against every node in the network using modified cosine similarity.

A match is accepted if:
- `score >= 0.7` (high confidence)
- `matched_peaks >= 6`

Candidates are ranked by score descending. The `mass_shift_da` field captures the precursor m/z difference between the unknown and the reference, which distinguishes exact matches from structural analogs.

### 3. Annotation Propagation

When no direct match is found (or as a complement), the unknown is anchored into the network and annotations are propagated through graph neighbors.

**Algorithm** (custom BFS-based propagation):

**Step 1 — Anchor selection.**
Scan all nodes. Any node with `similarity >= 0.5` and `matched_peaks >= 4` (more permissive thresholds than direct match) becomes an anchor — the query's entry point into the network.

**Step 2 — BFS traversal.**
From each anchor, perform a breadth-first search up to `max_depth=2` hops. The BFS frontier tracks:

```
(node_id, depth, cumulative_path_weight)
```

The cumulative path weight is the **product of edge cosine scores** along the traversal path.

**Step 3 — Propagation scoring.**
For each annotated node reached at depth `d`:

```
prop_score = anchor_similarity × Π(edge_weights along path) × decay_factor^d
```

Where:
- `anchor_similarity` — modified cosine between the query and the anchor node
- `Π(edge_weights)` — product of cosine scores along the path (penalizes weakly connected paths)
- `decay_factor^d` — exponential depth penalty (default: 0.7, so 0.70 at depth 1, 0.49 at depth 2)

**Step 4 — Aggregation.**
Compounds are deduplicated by SMILES (name as fallback). Only the best-scoring path per compound is kept. Compounds already found by direct matching are excluded.

**Relation to published methods:**
This algorithm is distinct from NAP (da Silva et al., *PLOS Comput. Biol.* 2018) and ChemWalker (Olivon et al., *Bioinformatics* 2023). Those methods use in silico fragmentation (MetFrag) and structural similarity re-ranking (NAP) or random walks (ChemWalker). This pipeline uses purely spectral and topological information — no in silico fragmentation is required. The scoring formula is a custom multiplicative path-weight scheme inspired by label propagation on weighted graphs.

---

## Annotation Decision Strategy

The `annotate` function returns a ranked DataFrame of candidates. The `select_annotation` function applies a decision rule to pick a single annotation per unknown.

### The three scenarios

**Scenario 1 — The unknown is in the reference library**

`source == "direct_match"`, `score >= 0.7`, `mass_shift_da < 0.02 Da`

The spectrum matches a library entry almost exactly. The top-1 compound is the annotation at **MSI level 2** (probable). No structural inference needed.

**Scenario 2 — The unknown is a structural analog**

Either:
- `source == "direct_match"` but `mass_shift_da >= 0.02 Da` → same fragmentation pattern, different molecular weight
- `source == "propagation"` → connected in the network but not directly matched

The top-1 candidate is the **closest known neighbor**, not the actual structure. The compound belongs to the same chemical class or homologous series. Annotation is **MSI level 3** (putative). See the *Structural Inference* section below.

**Scenario 3 — Truly novel compound**

Empty result DataFrame. No neighbors in the network, no anchor found. The compound lies outside the chemical space covered by the reference library. The pipeline cannot annotate it — external tools (SIRIUS/CSI:FingerID, CANOPUS) are required.

### Decision rule (`select_annotation`)

```
Direct match, mass_shift < 0.02 Da  →  "identified"     MSI-2
Direct match, mass_shift ≥ 0.02 Da  →  "analog of X"   MSI-3
Propagated, depth ≤ 1, score ≥ 0.3  →  "putative: X"   MSI-3
Otherwise                            →  None (unknown)
```

---

## Structural Inference for Analogs (Scenario 2)

When the unknown is an analog of a known compound B, two pieces of information can be extracted to infer the likely structure:

### Precursor mass shift → formula difference

The precursor m/z difference `Δm = mz_unknown - mz_reference` maps to a molecular formula delta. Common biochemical shifts:

| Δm (nominal) | Exact mass | Interpretation |
|---|---|---|
| +14 | +14.016 | +CH₂ (chain elongation) |
| +16 | +15.995 | +O (oxidation) |
| +28 | +27.995 (CO) or +28.031 (C₂H₄) | carbonylation or chain +2C |
| -18 | -18.011 | -H₂O (dehydration) |
| +2 | +2.016 | +H₂ (reduction) |
| +44 | +44.026 | +CO₂ (carboxylation) |

> **Note:** At unit mass resolution, 28 Da is ambiguous between CO (27.9949) and C₂H₄ (28.0313). High-resolution MS (Orbitrap, Q-TOF, ≤5 ppm) resolves these: they differ by 36 mDa. Always use the exact mass from the instrument when available.

### Fragment differential analysis → modification localization

Compare fragment ion lists of the unknown and the reference compound B:

- Fragments at the **same m/z** in both spectra → part of the **conserved scaffold** (modification is not in this substructure)
- Fragments shifted by **exactly Δm** in the unknown → contain the **modification site**
- Fragments unique to the unknown → arise from the modified region

This localization step constrains where on the scaffold the structural change occurred, significantly reducing the candidate space for structure enumeration.

### Proposed structure generation pipeline

```
unknown spectrum + best candidate SMILES
        ↓
Δm → molecular formula delta (lookup table)
        ↓
fragment differential analysis → modification localization
        ↓
enumerate modified structures with RDKit (apply Δm at plausible sites)
        ↓
in silico fragmentation of each candidate (CFM-ID or MetFrag)
        ↓
score simulated spectra vs unknown spectrum
        ↓
top-ranked modified structure = putative annotation (MSI-3)
```

This approach narrows the candidate space but does not eliminate ambiguity. The output remains a ranked list of structural proposals, not a confirmed identity. MSI level 3 is the ceiling without an authentic reference standard.

---

## MSI Confidence Levels

| Level | Label | Criteria |
|---|---|---|
| MSI-1 | Confirmed | Retention time + MS/MS match to authentic standard (not achievable with this pipeline alone) |
| MSI-2 | Probable | Direct match, `score >= 0.7`, `mass_shift < 0.02 Da` |
| MSI-3 | Putative | Propagated or analog match |
| MSI-4 | Unknown | No match or below threshold |

---

## Usage

### Build the molecular network

```python
from pipeline import build_molecular_network
from load_spectra import load_spectra

library_spectra = load_spectra("gnps_library.mgf")
network = build_molecular_network(library_spectra, path="net.json")
```

### Annotate unknowns

```python
from pipeline import annotate, select_annotation
import json
from store import json_to_graph

with open("net.json") as f:
    network = json_to_graph(json.load(f))

unknown_spectra = load_spectra("unknowns.mgf")

result_df = annotate(unknown_spectra[0], network)
decision = select_annotation(result_df)
print(decision)
# {"annotation": "putative: Quercetin", "confidence": "MSI-3", "type": "putative"}
```

### Batch annotation

```python
from pipeline import annotate_batch, export_results

results = annotate_batch(unknown_spectra, network, top_n=5)
export_results(results, "annotations.tsv", format="tsv")
```

---

## Dependencies

```
matchms
networkx
pandas
numpy
tqdm
```

---

## References

- Wang M. et al. *Sharing and community curation of mass spectrometry data with GNPS.* Nat. Biotechnol. 34, 828–837 (2016).
- Nothias L-F. et al. *Feature-based molecular networking in the GNPS analysis environment.* Nat. Methods 17, 905–908 (2020).
- da Silva R.R. et al. *Propagating annotations of molecular networks using in silico fragmentation.* PLOS Comput. Biol. 14, e1006089 (2018). [NAP]
- Olivon F. et al. *Improving annotation propagation on molecular networks through random walks: introducing ChemWalker.* Bioinformatics 39, btad078 (2023).
- Sumner L.W. et al. *Proposed minimum reporting standards for chemical analysis.* Metabolomics 3, 211–221 (2007). [MSI levels]
