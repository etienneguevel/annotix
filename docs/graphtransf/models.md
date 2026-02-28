# `annotix_ml/graphtransf/models/` — Core Neural Network Models

---

## `GnnNodeEdges` (`models/gnn.py`)

The **denoising network** for the DiGress diffusion process. A graph transformer that
takes a noisy graph `(h, e, y)` and predicts the clean graph logits `(pred_X, pred_E)`.

### Architecture

```
Input h (bs, n, natoms)           ← noisy node one-hot features
Input e (bs, n, n, nbonds)        ← noisy edge one-hot features
Input y (bs, dy)                  ← global features (extra features + normalised timestep)
Input node_features (bs, n, nf)   ← extra positional features (e.g. Laplacian eigenvectors)

   EmbeddingLaplacian   → project inputs to (d, de, dy)
         │
   ┌─────┴─────┐
   │ Attention  │  × n_layers
   │  Layer     │  (FiLM-modulated multi-head edge-node attention)
   └─────┬─────┘
         │
   MLPNodeEdge  →  h (bs, n, natoms), e (bs, n, n, nbonds)  ← logits
```

### Construction

```python
from annotix_ml.graphtransf.models.gnn import GnnNodeEdges

model = GnnNodeEdges(
    d=256,                # node hidden dimension
    de=64,                # edge hidden dimension
    dy=64,                # global hidden dimension
    n_heads=8,
    node_features=8,      # dim of extra node features (e.g. Laplacian eigenvectors)
    global_features=16,   # dim of extra global features (normalised timestep + cycle counts etc.)
    n_layers=5,
    natoms=17,            # number of atom types (VALID_ELEMENTS count)
    nbonds=5,             # number of bond types (TYPE_EDGES count)
    y_update=True,        # whether to update y at each attention layer
)
```

### Forward

```python
h, e, y, mask = model(e, mask, y, node_features, h)
```

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `e` | `(bs, n, n, nbonds)` | Noisy edge features (one-hot) |
| `mask` | `(bs, n)` | Binary mask — 1 for real atoms |
| `y` | `(bs, dy)` | Global conditioning vector (includes normalised timestep as last dim) |
| `node_features` | `(bs, n, node_features)` | Extra node-level positional features |
| `h` | `(bs, n, natoms)` | Noisy node features (one-hot) |

**Returns:** `(h, e, y, mask)` — `h` and `e` are the final logits over atom/bond types after
the `MLPNodeEdge` output head. The diffusion timestep is **not** passed directly — it is
embedded into `y` (as the normalised value `t / T`) by `compute_extra_features` before the
forward call.

### `GnnNodeEdgesWithoutY`

Variant without global conditioning. Constructor takes `(d, de, n_heads, node_features, global_features, n_layers, natoms, nbonds)` — no `dy` or `y_update`. The `global_features` are expanded to node shape and concatenated with `node_features` in the embedding layer.

```python
h, e, mask = model(e, mask, global_features, node_features, h)
```

---

## `NoisingModel` (`models/noising.py`)

Implements the **forward diffusion process** over discrete graph data following DiGress.

### Noise schedule

Uses a **cosine schedule** over `diffusion_steps` timesteps. The single-step transition
matrix from `t-1` to `t` is:

$$Q_t = \alpha_t \cdot I + (1 - \alpha_t) \cdot \mathbf{m}^{\top}$$

where $\alpha_t$ is the per-step retention fraction and $\mathbf{m}$ is the stationary
marginal distribution (dataset atom/bond frequencies). The cumulative matrix $\bar{Q}_t$
has the same form with $\bar{\alpha}_t$.

### Construction

```python
from annotix_ml.graphtransf.models.noising import NoisingModel

noiser = NoisingModel(
    nodes_distribution=train_dataset.nodes_distribution,  # (natoms,) atom type frequencies
    edges_distribution=train_dataset.edges_distribution,  # (nbonds,) bond type frequencies
    diffusion_steps=500,
    noise_schedule_type="cosine",  # only "cosine" is currently supported
)
```

Schedules (`alphas`, `alphas_bar`) and marginals are stored as plain tensor attributes.
Use `noiser.move_to(device)` to move them to the right device.

### Key methods

**`forward(N, E, node_mask, t=None) -> tuple[Tensor, Tensor, Tensor]`**

Applies forward diffusion. If `t` is not provided, a random timestep is sampled per graph.

```python
noised_N, noised_E, sampled_t = noiser(N, E, node_mask)
# noised_N:  (bs, n, natoms)    — one-hot noised node features
# noised_E:  (bs, n, n, nbonds) — one-hot noised edge features
# sampled_t: (bs, 1)            — timesteps used (returned for loss computation)
```

**`get_posterior(N, E, t) -> tuple[Tensor, Tensor]`**

Computes the posterior distribution $q(X_{t-1} \mid X_t, X_0)$ for all possible values of
$X_0$ simultaneously. Takes the **noised** graph at step `t` (not predicted `X_0`).

```python
pN_posterior, pE_posterior = noiser.get_posterior(N, E, t)
# pN_posterior: (bs, n, natoms, natoms)       — posterior over (X_0, X_{t-1}) pairs
# pE_posterior: (bs, n, n, nbonds, nbonds)    — posterior over (E_0, E_{t-1}) pairs
```

**`get_Q_t(t) -> tuple[Tensor, Tensor]`** / **`get_Q_bar_t(t) -> tuple[Tensor, Tensor]`**

Return the single-step `Q_t` and cumulative `Q̄_t` transition matrices for nodes and edges.

```python
Q_nodes, Q_edges = noiser.get_Q_bar_t(t)
# Q_nodes: (natoms, natoms), Q_edges: (nbonds, nbonds)
```

**`compute_noised_graph(N, E, t, node_mask=None) -> tuple[Tensor, Tensor]`**

Lower-level helper: applies `Q̄_t` and samples discrete features. Supports unbatched input.
