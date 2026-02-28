# `annotix_ml/graphtransf/math/` — Losses, Metrics, Extra Features, Noise Sampling

---

## `losses.py` — Training Loss

### `digress_loss(pN, pE, N, E, mask, loss_ratio, return_all=False)`

Cross-entropy loss for discrete graph diffusion. Computes the denoising loss over
both node (atom type) and edge (bond type) predictions.

```python
from annotix_ml.graphtransf.math.losses import digress_loss

# Default: returns scalar total loss
loss = digress_loss(pN, pE, N, E, mask, loss_ratio=1.0)

# With return_all=True: returns (total_loss, node_loss, edge_loss)
loss, node_loss, edge_loss = digress_loss(pN, pE, N, E, mask, loss_ratio=1.0, return_all=True)
# pN:  (bs, n, natoms)     — logits
# pE:  (bs, n, n, nedges)  — logits
# N:   (bs, n, natoms)     — one-hot ground truth
# E:   (bs, n, n, nedges)  — one-hot ground truth
# mask: (bs, n)
# total = node_loss + loss_ratio * edge_loss
```

Uses `CrossEntropyLoss` with `ignore_index=-100` to mask padded positions.

---

## `extra_features.py` — Graph Extra Features

Extra features are computed from the noised graph and concatenated to inputs before the
denoising network. Controlled by the `model.extra_features` config list.

### `laplacian_embedding(edges, k, mask=None) -> tuple[Tensor, Tensor]`

Computes the **k smallest non-trivial eigenvectors of the normalised Laplacian** plus
connected-component information.

```python
node_features, global_features = laplacian_embedding(edges, k=2, mask=mask)
# edges: (bs, n, n, nbonds)
# node_features:   (bs, n, k+1) — col 0: flag for largest connected component; cols 1..k: eigenvectors
# global_features: (bs, k+1)   — col 0: number of connected components; cols 1..k: eigenvalues (normalised by n)
```

### `node_cycle(edges, mask=None) -> tuple[Tensor, Tensor]`

Computes cycle membership counts for cycles of size 3, 4, 5 (node-level) and 3, 4, 5, 6 (graph-level).

```python
kcyclesx, kcyclesy = node_cycle(edges, mask=mask)
# edges: (bs, n, n, nbonds)
# kcyclesx: (bs, n, 3) — node-level counts for 3-, 4-, 5-cycles (clamped to [0,1])
# kcyclesy: (bs, 4)    — graph-level counts for 3-, 4-, 5-, 6-cycles (clamped to [0,1])
```

### `valency`, `charge`, `weight` — valence features

Three separate functions used to build the `"valence_features"` extra feature group:

```python
valency(edges, mask)                            # (bs, n, 1) — bond-weighted valence per atom
charge(nodes, edges, mask, valid_elements)      # (bs, n, 1) — formal charge (actual - covalent valence)
weight(nodes, valid_elements, max_weight=None)  # (bs, 1)    — total molecular weight, optionally normalised
```

### `spectra_fingerprint(num_peaks, types, instruments, ion_vec, form_vec, intens, spectra_encoder, projection) -> Tensor`

Encodes a batch of MS2 spectra into a fingerprint using `spectra_encoder` and projects the
output via `projection`. The encoder is called with `torch.no_grad()`.

```python
fp = spectra_fingerprint(
    num_peaks, types, instruments, ion_vec, form_vec, intens,
    spectra_encoder=encoder, projection=merge_fn,
)
# Returns (bs, projection_out_features)
```

---

## `noising.py` — Noise Schedule & Sampling

### `cosine_beta_schedule_discrete(timesteps, s=0.008) -> tuple[Tensor, Tensor]`

Computes the cosine noise schedule for `timesteps` diffusion steps.

```python
alphas, alphas_bar = cosine_beta_schedule_discrete(timesteps=500, s=0.008)
# Both tensors have shape (timesteps + 1,)
# alphas:     per-step retention fraction (1 - beta_t)
# alphas_bar: cumulative product of alphas (ᾱ_t)
```

The cumulative schedule follows:

$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\!\left(\frac{\tfrac{t}{\text{steps}} + s}{1 + s} \cdot \frac{\pi}{2}\right)^2$$

Betas are clamped to `[0, 0.9999]` to prevent instability; `alphas_bar` is then recomputed from the clamped values.

### `sample_discrete_features(probX, probE, node_mask) -> tuple[Tensor, Tensor]`

Samples discrete node and edge types from multinomial distributions. Edge samples are
upper-triangularised then symmetrised to enforce undirectedness.

```python
X_t, E_t = sample_discrete_features(
    probX,      # (bs, n, natoms)       — node type probabilities
    probE,      # (bs, n, n, nedges)    — edge type probabilities
    node_mask,  # (bs, n)
)
# Returns one-hot tensors: X_t (bs, n, natoms), E_t (bs, n, n, nedges)
```

Used during both forward diffusion (sampling `X_t`) and reverse generation (sampling `X_{t-1}`).

---

## `metrics.py` — Evaluation Metrics

### Molecular similarity

**`tanimoto_sim(smile1, smile2) -> float`**

Tanimoto coefficient between two molecules given as SMILES strings, computed from Morgan fingerprints (radius=3, nbits=2048).

**`MCES_distance(smile1, smile2) -> int`**

Maximum Common Edge Subgraph distance between two SMILES strings.
Defined as `(bonds_1 + bonds_2) - 2 * |MCES_bonds|`. Lower is more similar; 0 means identical.

### Batch metrics

**`compute_validity(pred_smiles) -> list[float]`**

Per-sample validity flag: `1.0` if SMILES is non-null, `0.0` otherwise.

**`compute_tanimoto_similarity(pred_smiles, true_smiles) -> list[float]`**

Per-sample Tanimoto similarity. Returns `0.0` for invalid or unparseable pairs.

**`compute_MCES_distance(pred_smiles, true_smiles) -> list[int]`**

Per-sample MCES distance. Returns `-1` for invalid or unparseable pairs.

**`compute_accuracy(pN, pE, N, E, mask) -> dict[str, list[float]]`**

Per-sample node and edge accuracy for a batch. Returns `{"node_accuracy": [...], "edge_accuracy": [...]}`.

**`compute_metrics(pred_smiles, true_smiles, pN, pE, N, E, mask) -> dict[str, list[float]]`**

Aggregates validity, Tanimoto similarity, and node/edge accuracy into a single dict.

### Training metrics

**`compute_training_metrics(pN, pE, N, E, mask, valid_elements) -> dict[str, float]`**

Computes scalar training diagnostics: per-atom binary cross-entropy and mean node/edge accuracy.

```python
metrics = compute_training_metrics(pN, pE, N, E, mask, valid_elements)
# metrics["ce_C"], metrics["ce_N"], ...   — per-atom-type binary CE
# metrics["node_accuracy"], metrics["edge_accuracy"]
```
