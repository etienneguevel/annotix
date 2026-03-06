# `annotix_ml/graphtransf/arch/` — Architecture Orchestrators

The arch module contains **orchestrator classes** that are *not* `nn.Module`. They wrap the
diffusion model components (`GnnNodeEdges` + `NoisingModel`), manage extra-feature computation,
and provide unified interfaces for training and generation.

---

## `DigressMetaArch` (`digress_meta_arch.py`)

The orchestrator for discrete graph diffusion, with optional fingerprint conditioning.

### Construction

Both class methods require dataset statistics extracted from the training set — these are
provided by `GraphDatasetFromSMILEs` attributes after the dataset is built.

**`init_from_cfg(cfg, device, valid_elements, nodes_distribution, edges_distribution, max_weight=None) -> DigressMetaArch`**

Builds all components from a config object and dataset statistics:
- Creates `GnnNodeEdges` or `GnnNodeEdgesWithoutY` from `cfg.model.*`
- Creates `NoisingModel` using `nodes_distribution` and `edges_distribution`
- If `cfg.get("spectra_encoder")` is set, reads `encoder_output_size` from
  `cfg.spectra_encoder.output_size` and builds a `fp_projector` accordingly
- Reads `morgan_nbits` from `cfg.dataset.morgan_nbits` (defaults to `0` if absent)
- Reads `fp_projector_type` from `cfg.model.fp_projector_type` (defaults to `"linear"`)

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

Calls `init_from_cfg` then loads weights from the checkpoint at `model_path` via
`load_checkpoint_state_dict`.

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
    "checkpoint": {
        "model":        <GnnNodeEdges state_dict>,
        "fp_projector": <nn.Linear or MLP state_dict>,  # only when fp_projector is active
    },
    "optimizer": <optimizer state_dict>,
}
```
Old-format checkpoints (bare state dict without the `"checkpoint"` envelope) are also supported.

---

### Key methods

**`forward_backward(nodes, edges, mask, fp=None) -> tuple[Tensor, Tensor, Tensor]`**

Runs the full training step for one batch:
1. Samples a random timestep `t` and noises the graph with `NoisingModel`
2. Computes extra features via `compute_extra_features`
3. If `fp` is provided, optionally projects it through `fp_projector` and appends to `y`
4. Runs the denoising network
5. Computes `digress_loss` and calls `.backward()`

Returns `(pN, pE, loss)`. All three are `None` on non-last ranks in pipeline parallelism.

```python
pN, pE, loss = arch.forward_backward(nodes, edges, mask)
# With fingerprint conditioning:
pN, pE, loss = arch.forward_backward(nodes, edges, mask, fp=fingerprint_tensor)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nodes` | `Tensor` | required | Node features `(bs, n, natoms)` |
| `edges` | `Tensor` | required | Edge features `(bs, n, n, nedges)` |
| `mask` | `Tensor` | required | Padding mask `(bs, n)` |
| `fp` | `Tensor \| None` | `None` | Pre-computed fingerprint `(bs, morgan_nbits)` |

---

**`generate(num_samples, max_nodes, min_nodes=6, progress_bar=False, num_attempts=3, fp=None) -> tuple[Tensor, Tensor, Tensor]`**

Runs the full reverse diffusion chain from pure noise. Returns raw graph tensors — SMILES
conversion is done by the caller via `batch_graph_to_smiles`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_samples` | `int \| Tensor` | required | Number of molecules to generate, or a 1D tensor of exact node counts per sample |
| `max_nodes` | `int` | required | Maximum number of nodes (padding size) |
| `min_nodes` | `int` | `6` | Minimum node count when sampling sizes randomly |
| `progress_bar` | `bool` | `False` | Show a tqdm denoising progress bar |
| `num_attempts` | `int` | `3` | Retry attempts on `LinAlgError` |
| `fp` | `Tensor \| None` | `None` | Pre-computed fingerprint `(bs, morgan_nbits)` passed to `forward` at each denoising step |

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

When `"spectra_fingerprint"` is in `extra_features`, it is **skipped** here. The fingerprint
contribution to `y` is injected by the callers (`forward`, `forward_backward`, `generate`),
which append the `fp` tensor (optionally projected through `fp_projector`) to `y` after this
method returns.

---

**`trainable_parameters() -> Iterator`**

Yields parameters from `diffuser` and, if active, `fp_projector`. Use this instead of
`diffuser.parameters()` to ensure all trainable components are included in the optimizer.

---

**`checkpoint_state_dict() -> dict`**

Returns the state dict for checkpointing:
```python
{"model": <GnnNodeEdges state_dict>}
# or, when fp_projector is active:
{"model": <GnnNodeEdges state_dict>, "fp_projector": <projector state_dict>}
```
Unwraps DDP modules automatically before extracting the state dict.

---

**`load_checkpoint_state_dict(state: dict)`**

Loads model state from a checkpoint state dict. Supports both the old format (bare
`GnnNodeEdges` state dict) and the new format (`{"model": ..., "fp_projector": ...}`).
Also restores `fp_projector` weights when present in the state dict.

---

**`sync_extra_gradients()`**

No-op in `DigressMetaArch` — DDP already wraps `self.diffuser`. Override in subclasses
that add trainable modules outside DDP to manually synchronize their gradients.

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
| `morgan_nbits` | `int` | Fingerprint vector size; contribution to `global_features` when `"spectra_fingerprint"` is active |
| `fp_projector` | `nn.Linear \| MLP \| None` | Optional projection from `encoder_output_size` → `morgan_nbits`; `None` when not configured |

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
  no_y: True                 # set False when using fingerprint conditioning
  diffusion_steps: 500
  noise_strategy: distribution
  num_ev: 2                  # number of Laplacian eigenvectors (for laplacian_embedding)
  fp_projector_type: linear  # "linear" or "mlp"; only used when spectra_fingerprint is active
  extra_features:
    - laplacian_embedding
    - node_cycle
    - valence_features
    - spectra_fingerprint    # enables fp conditioning; requires morgan_nbits in dataset config

dataset:
  morgan_nbits: 2048         # fingerprint size; required when spectra_fingerprint is active

# Optional: presence of this section causes init_from_cfg to read encoder_output_size
# and build an fp_projector mapping encoder outputs to morgan_nbits.
spectra_encoder:
  output_size: 4096
```
