# `annotix_ml/data/` — Datasets, Collators, Samplers

This submodule provides everything needed to load, represent, and batch molecular graphs — optionally paired with MS/MS spectra.

---

## `atoms_data.py` — Constants

Defines the vocabulary for graph node and edge features.

| Symbol | Type | Description |
|--------|------|-------------|
| `VALID_ELEMENTS` | `list[str]` | 17 supported atom types: C, N, O, S, P, F, Cl, Br, I, B, As, Si, Se, Fe, Co, K, Na |
| `DICT_EDGES` | `dict[str, int]` | Maps RDKit bond type names to integer indices |
| `TYPE_EDGES` | `list[str]` | Bond type vocabulary: `["NoBond", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]` |
| `ALLOWED_BONDS` | `dict[str, int]` | Maximum valence per element (e.g., C→4, N→3) |

---

## `dataset.py` — Graph Datasets

### `GraphDatasetMixin`

Mixin that converts SMILES strings into `(nodes, edges)` graph tensors.

- `smilesToGraph(smiles: str) -> tuple[Tensor, Tensor] | None` — Converts a SMILES to one-hot `(X, E)` tensors using `self.valid_elements` and `TYPE_EDGES`. Returns `None` for invalid SMILES or molecules containing atoms outside `valid_elements`.
- Hydrogens are implicit; edges are symmetrized; positions with no bond are one-hot encoded as `NoBond`.

### `GraphDatasetFromSMILEs`

`Dataset` for unconditional graph generation. Reads a **CSV file** (or a pandas DataFrame) with a column of SMILES strings.

```python
dataset = GraphDatasetFromSMILEs(
    data="data/train.csv",   # CSV path or DataFrame
    smile_column="smiles",   # column name containing SMILES
    split="train",           # optional: filter by this value in split_column
    split_column="split",    # optional: column used for split filtering
    max_nodes=50,
    cache_path="cache/train.pt",
)
nodes, edges = dataset[0]
# nodes: (n_atoms, len(valid_elements)), edges: (n_atoms, n_atoms, 5)
```

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `str \| DataFrame` | required | Path to a CSV file or a pandas DataFrame |
| `smile_column` | `str` | `"smiles"` | Name of the column containing SMILES strings |
| `split` | `str \| None` | `None` | Value to filter on in `split_column` (e.g. `"train"`) |
| `split_column` | `str \| None` | `None` | Column used for split filtering |
| `valid_elements` | `list[str] \| None` | `None` | Allowed atom symbols; auto-detected from data if `None` |
| `sanitizer` | `Callable \| None` | `None` | Extra filter applied to each `(nodes, edges)` graph |
| `verbose` | `bool` | `True` | Show progress bar during build |
| `cache_path` | `str \| None` | `None` | Path to a `.pt` cache; loaded if it exists, written after build |
| `save_cache` | `bool` | `True` | Whether to write the cache file (set `False` on non-main DDP ranks) |
| `max_nodes` | `int \| None` | `None` | Discard molecules with more atoms than this |

### The `sanitizer` argument

The `sanitizer` is an optional callable applied to every molecule after the graph is built from
SMILES. It runs during dataset construction (the `_build` phase) and is **not** called at
`__getitem__` time. A molecule is **kept** if the sanitizer returns a truthy value, and
**discarded** if it returns a falsy value (`None`, `False`, `0`, empty string, …).

**Signature:**

```python
def sanitizer(
    nodes: torch.Tensor,           # (n_atoms, len(valid_elements)) int one-hot
    edges: torch.Tensor,           # (n_atoms, n_atoms, 5) int one-hot
    valid_elements: list[str],     # atom vocabulary used to build the graph
) -> any  # truthy → keep, falsy → discard
```

**Built-in sanitizer — `graph_to_smiles_digress`**

All loaders in `loaders.py` pass `graph_to_smiles_digress` (from `data_utils.py`) as the
sanitizer. It:

1. Converts the graph back to an RDKit molecule
2. Attempts SMILES generation — returns `None` on valence or kekulization errors
3. If the molecule has multiple disconnected fragments, **keeps only the largest one** and
   returns its SMILES

This ensures every molecule in the final dataset is chemically valid and connected.

```python
from annotix_ml.data.data_utils import graph_to_smiles_digress

dataset = GraphDatasetFromSMILEs(
    data="data/train.csv",
    sanitizer=graph_to_smiles_digress,  # used by default in make_datasets()
)
```

**Writing a custom sanitizer:**

```python
def my_sanitizer(nodes, edges, valid_elements):
    # Example: reject molecules with fewer than 5 atoms
    n_atoms = nodes.shape[0]
    if n_atoms < 5:
        return None   # discard
    return True       # keep

dataset = GraphDatasetFromSMILEs(data="data/train.csv", sanitizer=my_sanitizer)
```

> **Note:** When a sanitizer is provided, the number of discarded molecules is printed at the
> end of the build step.

---

## `spec_dataset.py` — Graph + Spectra Dataset

### `GraphSpecDataset`

Paired dataset that returns `(graph, spectra)` items. Used in **spec2mol** training.

Requires a **CSV file** with SMILES and spectrum name columns. Spectra are read as `.ms` files
from `spec_folder`; subformula annotations are read from `subform_folder`.

```python
dataset = GraphSpecDataset(
    data="data/paired.csv",          # CSV with smiles + spectrum name columns
    spec_folder="data/spectra/",     # folder containing <spec_name>.ms files
    subform_folder="data/subforms/", # folder with subformula JSON annotations
    smile_column="smiles",
    spec_column="spec",
    max_nodes=50,
)
item = dataset[0]
# item["nodes"], item["edges"], item["smiles"], item["spec_name"],
# item["peak_type"], item["form_vec"], item["ion_vec"],
# item["frag_intens"], item["instrument"], item["magma_fps"], ...
```

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `str \| DataFrame` | required | Path to a CSV/TSV file or a pandas DataFrame |
| `spec_folder` | `str \| Path` | required | Folder containing `<spec_name>.ms` spectrum files |
| `subform_folder` | `str \| Path` | required | Folder with subformula JSON annotations for each spectrum |
| `smile_column` | `str` | `"smiles"` | Column name for SMILES strings |
| `spec_column` | `str` | `"spec"` | Column name for spectrum identifiers |
| `formula_column` | `str` | `"formula"` | Column name for molecular formula |
| `instrument_column` | `str` | `"instrument"` | Column name for instrument type |
| `valid_elements` | `list[str] \| None` | `None` | Allowed atom symbols; auto-detected if `None` |
| `sanitizer` | `Callable \| None` | `None` | Extra filter applied to each `(nodes, edges)` graph |
| `verbose` | `bool` | `True` | Show progress bar during build |
| `cache_path` | `str \| None` | `None` | Path to a `.pt` cache; loaded if it exists, written after build |
| `save_cache` | `bool` | `True` | Whether to write the cache file |
| `max_nodes` | `int \| None` | `None` | Discard molecules with more atoms than this |
| `**kwargs` | | | Extra kwargs forwarded to `PeakFormula` (the spectra featurizer) |

**Item dict keys** (from `__getitem__`):

| Key | Description |
|-----|-------------|
| `nodes` | `(n_atoms, len(valid_elements))` one-hot atom tensor |
| `edges` | `(n_atoms, n_atoms, 5)` one-hot bond tensor |
| `smiles` | SMILES string |
| `spec_name` | Spectrum identifier |
| `peak_type` | Peak type indices array |
| `form_vec` | Subformula element count vectors array |
| `ion_vec` | Ion mode index per peak |
| `frag_intens` | Peak intensities array |
| `instrument` | Instrument type index |
| `magma_fps` | Fragment fingerprints from MAGMA (zeros if unavailable) |

---

## `datacollator.py` — Collate Functions

All collators are designed to be passed to `torch.utils.data.DataLoader` as `collate_fn`.

### `collateGraph(batch)`

Dynamic-padding collator for `GraphDatasetFromSMILEs`. Pads all graphs in a batch to the maximum number of atoms in that batch.

```python
loader = DataLoader(dataset, batch_size=32, collate_fn=collateGraph)
nodes, edges, mask = next(iter(loader))
# nodes: (32, max_n, 17), edges: (32, max_n, max_n, 5), mask: (32, max_n)
```

**Input:** list of `(nodes, edges)` 2-tuples.
**Output:** `(nodes, edges, mask)` where `mask[i, j] = 1` if atom `j` is real in molecule `i`.

---

### `collateGraphStatic(batch, n_max)`

Like `collateGraph` but pads to a **fixed** `n_max` regardless of batch content. Required for pipeline parallelism where static shapes are needed.

```python
from functools import partial
collate_fn = partial(collateGraphStatic, n_max=50)
```

---

### `collateGraphJagged(batch)`

Experimental jagged-tensor collator. Stacks all graphs into one flat `(1, L, d)` tensor and computes a `FlexAttention` block mask. Not used in the main training pipeline.

---

### `collateGraphSpec(batch)`

Joint graph + spectra collator for `GraphSpecDataset`. Pads graphs to the batch maximum and delegates spectra collation to `PeakFormula.collate_fn`.

```python
loader = DataLoader(spec_dataset, batch_size=16, collate_fn=collateGraphSpec)
nodes, edges, mask, num_peaks, types, instruments, ion_vec, form_vec, intens = next(iter(loader))
```

**Output tuple (9 elements):**

| Index | Name | Shape | Description |
|-------|------|-------|-------------|
| 0 | `nodes` | `(bs, max_n, 17)` | Padded atom one-hot features |
| 1 | `edges` | `(bs, max_n, max_n, 5)` | Padded bond one-hot features |
| 2 | `mask` | `(bs, max_n)` | Bool mask — True for real atoms |
| 3 | `num_peaks` | `(bs,)` | Number of peaks per spectrum |
| 4 | `types` | `(bs, max_peaks)` | Peak type indices |
| 5 | `instruments` | `(bs,)` | Instrument type indices |
| 6 | `ion_vec` | `(bs, ion_dim)` | Ionization mode embedding |
| 7 | `form_vec` | `(bs, form_dim)` | Molecular formula embedding |
| 8 | `intens` | `(bs, max_peaks)` | Peak intensities |

---

## `samplers.py` — Infinite Sampler

### `InfiniteSampler`

A `Sampler` that yields indices indefinitely, supporting:
- **Shuffling** — deterministic per-epoch shuffle via a seeded `torch.Generator`
- **Distributed training** — strides over the index range by `step`, starting at `start`, so each rank sees a disjoint subset
- **Resume** — `advance` skips the first N indices yielded, allowing training to resume exactly where it left off

```python
sampler = InfiniteSampler(
    sample_count=len(dataset),  # total number of samples in the dataset
    shuffle=True,
    seed=42,
    start=0,     # index offset — typically the global rank
    step=4,      # stride — typically the world size
    advance=1000,  # skip first 1000 indices (resume from checkpoint)
)
loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

**Constructor parameters** (all keyword-only):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sample_count` | `int` | required | Total number of samples in the dataset |
| `shuffle` | `bool` | `False` | Whether to shuffle indices each epoch |
| `seed` | `int` | `0` | Random seed for reproducible shuffling |
| `start` | `int \| None` | `None` | Starting offset into the index range; auto-reads `get_global_rank()` if `None` |
| `step` | `int \| None` | `None` | Stride between consecutive indices for this rank; auto-reads `get_global_size()` if `None` |
| `advance` | `int` | `0` | Number of indices to skip at the start of iteration (for checkpoint resume) |

When `start` and `step` are both `None`, the sampler automatically reads the global rank and
world size from the process group, so no explicit distributed arguments are needed after
`dist.enable()` has been called.

**Usage in `train.py`:**

```python
data_rank = dist.get_global_rank()
data_size = dist.get_global_size()

sampler = InfiniteSampler(
    sample_count=len(train_dataset),
    shuffle=True,
    seed=cfg.train.seed,
    start=data_rank,
    step=data_size,
    advance=advance,   # (last_epoch + 1) * batch_size % len(dataset)
)
```

---

## `loaders.py` — Dataset Factory

### `make_datasets(...) -> tuple[GraphDatasetFromSMILEs, GraphDatasetFromSMILEs]`

Builds train and validation `GraphDatasetFromSMILEs` objects from a single CSV file,
splitting on a fold column. Returns the two *datasets* — `DataLoader` wrapping is done
by the training code.

```python
from annotix_ml.data.loaders import make_datasets

train_dataset, valid_dataset = make_datasets(
    data_path="data/molecules.csv",
    smile_column="smiles",    # default
    split_column="fold",      # default — column used to split train vs val
    val_tag="test",           # default — value in split_column that marks val rows
    verbose=True,
    cache_path="cache/mol",   # produces cache/mol_train.pt and cache/mol_valid.pt
    save_cache=True,
    max_nodes=50,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_path` | `str` | required | Path to the CSV file |
| `smile_column` | `str` | `"smiles"` | Column containing SMILES strings |
| `split_column` | `str` | `"fold"` | Column used to split train vs validation |
| `val_tag` | `str \| None` | `"test"` | Value in `split_column` that identifies validation rows; all other rows become train |
| `verbose` | `bool` | `True` | Show progress bar during build |
| `cache_path` | `str \| None` | `None` | Base path for cache files; `_train.pt` and `_valid.pt` are appended |
| `save_cache` | `bool` | `True` | Whether to write cache files after building |
| `max_nodes` | `int \| None` | `None` | Discard molecules with more atoms than this |

`graph_to_smiles_digress` is always used as the sanitizer.

---

### `make_spec_datasets(cfg, save_cache, verbose) -> tuple[GraphSpecDataset, GraphSpecDataset]`

Builds train and validation `GraphSpecDataset` objects from a config object.
Splits the CSV at `cfg.dataset.data_path` on the `cfg.dataset.split_column` column,
using `cfg.dataset.val_tag` to identify validation rows. Returns two *datasets*.

The `valid_elements` vocabulary is derived from the train split and reused for the
validation split to ensure consistency.

```python
from annotix_ml.data.loaders import make_spec_datasets

train_dataset, valid_dataset = make_spec_datasets(cfg, save_cache=True, verbose=True)
```

**Config keys read:**

| Key | Description |
|-----|-------------|
| `cfg.dataset.data_path` | Path to the CSV/TSV file |
| `cfg.dataset.split_column` | Column used for train/val split |
| `cfg.dataset.val_tag` | Value in `split_column` that identifies validation rows |
| `cfg.dataset.spec_folder` | Folder containing `.ms` spectrum files |
| `cfg.dataset.subform_folder` | Folder containing subformula annotations |
| `cfg.dataset.smile_column` | Column containing SMILES strings |
| `cfg.dataset.cache_path` | Base path for cache files (optional) |
| `cfg.dataset.max_nodes` | Maximum number of atoms per molecule (optional) |

`graph_to_smiles_digress` is always used as the sanitizer.

---

## `data_utils.py` — Graph Utilities

### `mask_any_tensor(t, mask, fill_value)`

Applies a node mask to a graph tensor, setting masked positions to `fill_value`.

The function expects ``mask`` to have shape equal to the leading (left-most)
dimensions of ``t``. Any remaining trailing dimensions of ``t`` are treated
as feature dimensions and the mask is expanded (unsqueezed) over them.

```python
# Zero out padded nodes
nodes = mask_any_tensor(nodes, mask, fill_value=0.0)
edges = mask_any_tensor(edges, mask, fill_value=0.0)
```

### `graph_to_smiles(nodes, edges, valid_elements) -> str | None`

Converts a single `(nodes, edges)` graph tensor to a SMILES string via RDKit.
Returns `None` if the graph is chemically invalid.

### `batch_graph_to_smiles(nodes, edges, mask, valid_elements) -> list[str | None]`

Batch version of `graph_to_smiles`. Applies the mask to extract real atoms before conversion.

### `graph_to_smiles_digress` / `batch_graph_to_smiles_digress`

Variants that apply DiGress-specific post-processing (argmax over one-hot, bond symmetrization) before calling RDKit.
