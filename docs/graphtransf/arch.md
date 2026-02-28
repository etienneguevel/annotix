# `annotix_ml/graphtransf/arch/` — Architecture Orchestrators

The arch module contains **orchestrator classes** that are *not* `nn.Module`. They wrap the
diffusion model components (`GnnNodeEdges` + `NoisingModel`), manage extra-feature computation,
and provide unified interfaces for training and generation.

---

## `DigressMetaArch` (`digress_meta_arch.py`)

The base orchestrator for unconditional discrete graph diffusion.

### Construction

Both class methods require dataset statistics extracted from the training set — these are
provided by `GraphDatasetFromSMILEs` attributes after the dataset is built.

**`init_from_cfg(cfg, device, valid_elements, nodes_distribution, edges_distribution, max_weight=None) -> DigressMetaArch`**

Builds all components from a config object and dataset statistics:
- Creates `GnnNodeEdges` or `GnnNodeEdgesWithoutY` from `cfg.model.*`
- Creates `NoisingModel` using `nodes_distribution` and `edges_distribution`

```python
arch = DigressMetaArch.init_from_cfg(
    cfg,
    device=torch.device("cuda"),
    valid_elements=train_dataset.valid_elements,
    nodes_distribution=train_dataset.nodes_distribution,
    edges_distribution=train_dataset.edges_distribution,
    max_weight=train_dataset.max_weight,
)
```

**`load_pretrained(cfg, device, model_path, valid_elements, nodes_distribution, edges_distribution, max_weight=None) -> DigressMetaArch`**

Calls `init_from_cfg` then loads `diffuser` weights from the checkpoint at `model_path`.

```python
arch = DigressMetaArch.load_pretrained(
    cfg,
    device=torch.device("cuda"),
    model_path="logs/experiment/10000.pt",
    valid_elements=train_dataset.valid_elements,
    nodes_distribution=train_dataset.nodes_distribution,
    edges_distribution=train_dataset.edges_distribution,
    max_weight=train_dataset.max_weight,
)
```

The checkpoint file is expected in the format saved by `checkpoint_state_dict()`:
```python
{
    "checkpoint": {"model": <GnnNodeEdges state_dict>},
    "optimizer": <optimizer state_dict>,
}
```
Old-format checkpoints (bare state dict without the `"checkpoint"` envelope) are also supported.

---

### Key methods

**`forward_backward(nodes, edges, mask, *args) -> tuple[Tensor, Tensor, Tensor]`**

Runs the full training step for one batch:
1. Samples a random timestep `t` and noises the graph with `NoisingModel`
2. Computes extra features via `compute_extra_features`
3. Runs the denoising network
4. Computes `digress_loss` and calls `.backward()`

Returns `(pN, pE, loss)`. All three are `None` on non-last ranks in pipeline parallelism.

```python
pN, pE, loss = arch.forward_backward(nodes, edges, mask)
# For spec2mol: arch.forward_backward(nodes, edges, mask, num_peaks, types, instruments, ion_vec, form_vec, intens)
```

---

**`generate(num_samples, max_nodes, min_nodes=6, progress_bar=False, num_attempts=3, other_args=[]) -> tuple[Tensor, Tensor, Tensor]`**

Runs the full reverse diffusion chain from pure noise. Returns raw graph tensors — SMILES
conversion is done by the caller via `batch_graph_to_smiles`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_samples` | `int \| Tensor` | required | Number of molecules to generate, or a 1D tensor of exact node counts per sample |
| `max_nodes` | `int` | required | Maximum number of nodes (padding size) |
| `min_nodes` | `int` | `6` | Minimum node count when sampling sizes randomly |
| `progress_bar` | `bool` | `False` | Show a tqdm denoising progress bar |
| `num_attempts` | `int` | `3` | Retry attempts on `LinAlgError` |
| `other_args` | `list` | `[]` | Extra args forwarded to `compute_extra_features` (e.g. spectra tensors) |

Returns `(N, E, mask)` — graph tensors of shape `(bs, max_nodes, natoms)`, `(bs, max_nodes, max_nodes, nedges)`, `(bs, max_nodes)`.

```python
N, E, mask = arch.generate(num_samples=64, max_nodes=40, progress_bar=True)
smiles = batch_graph_to_smiles(N, E, mask, arch.valid_elements)
```

---

**`compute_extra_features(nodes, edges, mask, t) -> tuple[Tensor, Tensor]`**

Computes extra features for the current (noised) graph state and returns:
- `pos_emb` — node-level positional features `(bs, n, node_feat_dim)`
- `y` — global conditioning vector `(bs, global_feat_dim)`, always including the normalised
  timestep `t / T` as the last dimension

Features computed depend on `cfg.model.extra_features`:
- `"laplacian_embedding"` — Laplacian eigenvectors + connected component count (node + global)
- `"node_cycle"` — cycle membership for rings of size 3–5 (node + global)
- `"valence_features"` — valency + charge (node), normalised molecular weight (global)

---

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `diffuser` | `GnnNodeEdges \| GnnNodeEdgesWithoutY` | The denoising network |
| `noiser` | `NoisingModel` | The forward diffusion model |
| `valid_elements` | `list[str]` | Atom type vocabulary |
| `extra_features` | `list[str]` | Names of active extra features |
| `device` | `torch.device` | Device the model lives on |
| `loss_ratio` | `float` | Edge loss weight relative to node loss |

---

## `Spec2MolMetaArch` (`spec2mol_meta_arch.py`)

Extends `DigressMetaArch` with spectrum conditioning. Adds a `SpectraEncoderGrowing` that
maps MS2 spectra to a fingerprint vector appended to the global conditioning `y`.

### Construction

**`init_from_cfg(cfg, device, valid_elements, nodes_distribution, edges_distribution, max_weight=None) -> Spec2MolMetaArch`**

Calls `DigressMetaArch.init_from_cfg`, then:
- Builds `SpectraEncoderGrowing` from `cfg.spectra_encoder.*`
- If `cfg.spectra_encoder.checkpoint_path` is set, loads pretrained encoder weights and
  freezes the encoder in eval mode
- Creates `merge_function`: `nn.Linear(output_size, morgan_nbits)` where `morgan_nbits`
  comes from `cfg.dataset.morgan_nbits`

```python
arch = Spec2MolMetaArch.init_from_cfg(
    cfg,
    device=torch.device("cuda"),
    valid_elements=train_dataset.valid_elements,
    nodes_distribution=train_dataset.nodes_distribution,
    edges_distribution=train_dataset.edges_distribution,
    max_weight=train_dataset.max_weight,
)
```

**`load_pretrained`** inherits from `DigressMetaArch` and additionally restores
`merge_function` weights from the `"merge_function"` key in the checkpoint.

### Extra attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `spectra_encoder` | `SpectraEncoderGrowing` | Encodes MS2 peaks → fingerprint `(bs, output_size)` |
| `merge_function` | `nn.Linear` | Projects fingerprint `(output_size → morgan_nbits)` |

### Checkpoint format

```python
{
    "checkpoint": {
        "model":          <GnnNodeEdges state_dict>,
        "merge_function": <nn.Linear state_dict>,
    },
    "optimizer": <optimizer state_dict>,
}
```

Note: `spectra_encoder` weights are **not** saved in the training checkpoint — they are loaded
separately at init time via `cfg.spectra_encoder.checkpoint_path`.

### `compute_extra_features` (overridden)

Delegates standard features to `DigressMetaArch.compute_extra_features`, then — if
`"spectra_fingerprint"` is in `extra_features` — encodes the spectra batch through
`spectra_encoder` and `merge_function`, and appends the resulting fingerprint to `y`:

```python
y = torch.cat([y_from_parent, merge_function(spectra_encoder(spectra_batch))], dim=-1)
```

### Spectrum-conditioned generation

There is no separate `generate_from_spectra` method. Conditioning is passed via the
`other_args` parameter of the standard `generate` call:

```python
# spectra_args = [num_peaks, types, instruments, ion_vec, form_vec, intens]
N, E, mask = arch.generate(
    num_samples=n,       # 1D tensor of node counts matching the batch
    max_nodes=n.max(),
    other_args=spectra_args,
)
```

---

## Configuration reference

```yaml
model:
  d: 256                     # node hidden dimension
  de: 64                     # edge hidden dimension
  dy: 64                     # global (y) hidden dimension
  n_heads: 8
  n_layers: 5
  y_update: False
  no_y: True                 # set False when using spectra conditioning
  diffusion_steps: 500
  noise_strategy: distribution
  num_ev: 2                  # number of Laplacian eigenvectors (for laplacian_embedding)
  extra_features:
    - laplacian_embedding
    - node_cycle
    - valence_features
    - spectra_fingerprint    # Spec2Mol only

spectra_encoder:             # presence of this section activates Spec2MolMetaArch
  form_embedder: pos-cos
  output_size: 4096
  hidden_size: 512
  spectra_dropout: 0
  top_layers: 1
  refine_layers: 4
  magma_modulo: 2048
  peak_attn_layers: 2
  set_pooling: intensity
  pairwise_featurization: True
  num_heads: 8
  embed_instrument: False
  inten_transform: float
  no_diffs: False
  checkpoint_path: checkpoints/encoder.pt   # optional pretrained encoder weights
```
