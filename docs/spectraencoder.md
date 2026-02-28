# `annotix_ml/spectraencoder/` — MS/MS Spectra Encoder

The spectraencoder submodule processes **tandem mass spectrometry (MS2)** spectra and encodes them
into dense fingerprint vectors suitable as conditioning inputs for graph generation.

This submodule was recovered from the [DiffMs](https://github.com/coleygroup/DiffMS/tree/master)
repo, and was located within the `mist` submodule.

---

## `model/spectra_encoder.py` — Top-level Encoder Models

### `SpectraEncoder`

`nn.Module` that combines a `FormulaTransformer` peak encoder with a final fingerprint prediction head.

```python
from annotix_ml.spectraencoder.model.spectra_encoder import SpectraEncoder

encoder = SpectraEncoder(
    form_embedder="float",    # formula embedding type
    output_size=4096,         # fingerprint output size
    hidden_size=50,
    peak_attn_layers=2,
    top_layers=1,
    magma_modulo=2048,
    set_pooling="intensity",
    pairwise_featurization=False,
    num_heads=8,
    embed_instrument=False,
    inten_transform="float",
    no_diffs=False,
)

fp, aux = encoder(batch)
# fp: (bs, output_size) — Morgan-like fingerprint in [0, 1]
# aux["pred_frag_fps"]: (bs, max_peaks, magma_modulo) — per-peak fragment fingerprints
# aux["h0"]: (bs, hidden_size) — pooled hidden state before the output head
```

**`init_from_cfg(cfg: DictConfig) -> SpectraEncoder`**

Constructs the encoder from `cfg.spectra_encoder.*`.

**Forward pass:**
1. `FormulaTransformer` encodes peaks → pooled hidden state `h` of shape `(bs, hidden_size)`
2. `fragment_predictor` MLP: `h_peaks → (bs, max_peaks, magma_modulo)` fragment fingerprints
3. `spectra_predictor` MLP + Sigmoid: `h → (bs, output_size)` spectral fingerprint

---

### `SpectraEncoderGrowing`

Variant of `SpectraEncoder` that replaces the final linear head with `FPGrowingModule` — a
hierarchical multi-resolution output that progressively refines the fingerprint.

```python
encoder = SpectraEncoderGrowing(
    output_size=4096,
    refine_layers=3,   # number of intermediate refinement steps
    # ... other params same as SpectraEncoder
)

fp, aux = encoder(batch)
# fp: (bs, output_size) — final fingerprint
# aux["int_preds"]: list of (bs, output_size/2^k) intermediate fingerprints
# aux["pred_frag_fps"]: (bs, max_peaks, magma_modulo)
```

**`init_from_cfg(cfg: DictConfig) -> SpectraEncoderGrowing`**

Same as `SpectraEncoder.init_from_cfg` but additionally reads `cfg.spectra_encoder.refine_layers`.

---

## `model/modules.py` — Internal Modules

### `FormulaTransformer`

The core spectrum encoder. Processes a bag of `(m/z, intensity, type)` peaks using
multi-head self-attention with a learnable formula embedding.

```python
transformer = FormulaTransformer(
    hidden_size=50,
    peak_attn_layers=2,
    set_pooling="intensity",   # "intensity" or "mean" or "cls"
    output_size=4096,
    form_embedder="float",
    num_heads=8,
)
h, aux = transformer(batch, return_aux=True)
# h: (bs, hidden_size) — pooled spectrum representation
# aux["peak_tensor"]: (bs, max_peaks, hidden_size) — per-peak embeddings
```

**Set pooling strategies:**
- `"intensity"` — weighted average of peak embeddings by intensity
- `"mean"` — simple average
- `"cls"` — use a learned CLS token

### `FPGrowingModule`

Hierarchical fingerprint decoder. Builds the output fingerprint at multiple resolutions:
each refinement level doubles the output size.

```python
module = FPGrowingModule(
    hidden_input_dim=50,
    final_target_dim=4096,
    num_splits=3,      # 3 intermediate steps
    reduce_factor=2,   # each step halves resolution
)
outputs = module(h)   # list of tensors: [(bs, 512), (bs, 1024), (bs, 2048), (bs, 4096)]
```

Returns a list where the last element is the final fingerprint.

### `MLPBlocks`

Simple multi-layer MLP with optional LayerNorm and ReLU activations. Used internally.

---

## `model/form_embedders.py` — Formula Embedders

These modules embed the molecular formula (empirical formula of the precursor ion) into a vector
that conditions the `FormulaTransformer`.

### `FloatFeaturizer`

Represents each formula as raw integer counts (C, H, N, O, S, ...) cast to float.

### `IntFeaturizer`

Embeds each element count as a learned integer embedding, then concatenates.

### `FourierFeaturizer`

Applies Fourier feature encoding to element counts for richer frequency-domain representation.

### `get_embedder(form_embedder: str, hidden_size: int) -> nn.Module`

Factory function. Accepts `"float"`, `"int"`, or `"fourier"`.

```python
embedder = get_embedder("float", hidden_size=50)
formula_emb = embedder(form_vec)  # (bs, hidden_size)
```

---

## `data/objects.py` — Data Classes

### `Spectra`

Represents a single MS2 spectrum from a `.ms` file.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Spectrum name / compound ID |
| `formula` | `str` | Molecular formula string |
| `smiles` | `str \| None` | SMILES if available |
| `peaks` | `list[tuple[float, float]]` | List of `(m/z, intensity)` pairs |
| `ion_mode` | `str` | `"positive"` or `"negative"` |
| `instrument` | `str` | Instrument type string |

### `Mol`

Container for a molecule with SMILES and optional precomputed Morgan fingerprint.

```python
mol = Mol(smiles="CC(=O)O", fp=morgan_fp)
```

---

## `data/featurizers.py` — Batch Featurizers

### `PeakFormula`

Featurizer that converts `Spectra` objects into padded tensor batches ready for `FormulaTransformer`.

**`featurize(spectra: Spectra) -> dict`**

Converts a single spectrum to a dict with keys:
- `mzs` — `(n_peaks,)` m/z values
- `intens` — `(n_peaks,)` normalized intensities
- `types` — `(n_peaks,)` peak type indices
- `ion_vec` — ionization mode one-hot
- `form_vec` — formula element count vector
- `instrument` — instrument type index

**`collate_fn(batch: list[dict]) -> dict`**

Pads a list of featurized spectra to the same length and returns a batch dict with keys:
`num_peaks`, `types`, `instruments`, `ion_vec`, `form_vec`, `intens`.

```python
featurizer = PeakFormula()
item = featurizer.featurize(spectrum)
batch = PeakFormula.collate_fn([item1, item2, item3])
```

### `FingerprintFeaturizer`

Computes Morgan fingerprints for molecules using RDKit.

```python
ff = FingerprintFeaturizer(radius=2, nbits=2048)
fp = ff.featurize(smiles)  # numpy array of shape (2048,)
```

### `PairedFeaturizer`

Combines `PeakFormula` and `FingerprintFeaturizer` to produce `(spectrum_features, fingerprint)` pairs for training the spectra encoder in supervised mode.

---

## `data/splitter.py` — Train/Val/Test Split

### `SpectraSplitter`

Abstract base class for splitting a list of `Spectra` objects.

### `PresetSpectraSplitter`

Uses preset molecule/spectrum IDs to assign each item to train/val/test.

```python
splitter = PresetSpectraSplitter(
    train_ids={"mol_001", "mol_002", ...},
    val_ids={"mol_101", ...},
    test_ids={"mol_201", ...},
)
train, val, test = splitter.split(spectra_list)
```

---

## `utils.py` — Spectrum Utilities

### `parse_spectra(path: str) -> list[Spectra]`

Parses a `.ms` file into a list of `Spectra` objects.

```python
spectra = parse_spectra("data/train.ms")
```

### `formula_to_dense(formula: str) -> np.ndarray`

Converts an empirical formula string (e.g., `"C6H12O6"`) to a fixed-length integer array
over the standard element vocabulary.

```python
arr = formula_to_dense("C6H12O6")  # array([6, 12, 6, 0, 0, ...])
```

### `ION_LST`

List of recognized ionization mode strings (e.g., `"[M+H]+"`).

### `PlaceHolder`

`NamedTuple` carrying the graph state `(X, E, y, mask)` through the model pipeline.
Provides a clean container to pass graph states between modules.
