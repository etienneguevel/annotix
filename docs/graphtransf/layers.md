# `annotix_ml/graphtransf/layers/` — Neural Network Layers

Building blocks used by `GnnNodeEdges` and the broader graph transformer architecture.

---

## `embeddings.py` — Input Projections

### `EmbeddingLaplacian`

Projects raw graph inputs into the transformer's hidden dimensions via independent `MLP` modules for nodes, edges, global features, and extra node features (e.g. Laplacian positional encodings). Node output is the sum of the node-type embedding and the positional embedding.

```python
emb = EmbeddingLaplacian(
    d=256,               # node hidden dim
    de=64,               # edge hidden dim
    dy=64,               # global hidden dim
    node_features=8,     # dim of extra node features (e.g. Laplacian eigenvectors)
    global_features=16,  # dim of extra global features (e.g. normalised timestep + cycle counts)
    natoms=17,           # number of node types
    nbonds=5,            # number of edge types
)
h, e, y, mask = emb(N, E, global_features_t, node_features_t, mask)
# h: (bs, n, d), e: (bs, n, n, de), y: (bs, dy)
```

### `EmbeddingLaplacianWithoutY`

Variant used with `GnnNodeEdgesWithoutY`. No `dy` or `global_features` constructor args — instead, `global_features` are passed at forward time, expanded to node shape `(bs, n, global_features_dim)`, and concatenated with `node_features_extra` before the `LaplacianProjection` MLP. Therefore `node_features` must equal `node_features_extra_dim + global_features_dim`.

```python
emb = EmbeddingLaplacianWithoutY(
    d=256,
    de=64,
    node_features=24,  # must equal node_features_extra_dim + global_features_dim
    natoms=17,
    nbonds=5,
)
h, e, mask = emb(N, E, global_features, node_features_extra, mask)
# h: (bs, n, d), e: (bs, n, n, de) — no y output
```

---

## `attention.py` — Graph Attention

### `MultiHeadEdgeNodeWithY`

The core attention block. Performs **FiLM-modulated multi-head attention** over both node and edge features, with optional global conditioning from `y`.

```python
attn = MultiHeadEdgeNodeWithY(
    d=256,         # node hidden dimension
    de=64,         # edge hidden dimension
    dy=64,         # global feature dimension
    n_heads=8,
    y_update=True, # whether to update y (default True)
)
h_out, e_out, y_out, mask = attn(h, e, y, mask)
# h_out: (bs, n, d), e_out: (bs, n, n, de), y_out: (bs, dy)
```

#### `compute_attn(h, e, y, mask, attn_map_mode=False)`

The full attention computation. Returns `(node_attn, edge_attn)` where `node_attn` is `(bs, n, d)` and `edge_attn` is `(bs, n, n, d)` — note `d` not `de`; the projection back to `de` happens in `forward` via `Out_E`.

**Step-by-step computation:**

**1. QKV projection**

A single `Linear(d, 3d)` projects node features, then the result is split across heads:

$$Q, K, V \in \mathbb{R}^{bs \times n_h \times n \times d_k}, \quad d_k = d / n_h$$

**2. Edge FiLM projection**

Edge features are projected to two `dk`-dim tensors per head via `Linear(de, 2d)`:

$$E_1, E_2 \in \mathbb{R}^{bs \times n_h \times n \times n \times d_k}$$

**3. Raw attention scores**

Rather than a dot product, Q and K are multiplied **element-wise** after broadcasting — each of the `dk` dimensions is kept separately:

$$A_{ij} = \frac{Q_i \odot K_j}{\sqrt{d_k}} \in \mathbb{R}^{bs \times n_h \times n \times n \times d_k}$$

**4. Edge-modulated attention (FiLM)**

The `dk`-dim attention tensor is modulated by the projected edge features:

$$A'_{ij} = E_{1,ij} + E_{2,ij} \odot A_{ij} + A_{ij} = E_{1,ij} + (1 + E_{2,ij}) \odot A_{ij}$$

**5. Scalar attention weights (sum-collapse + softmax)**

The `dk` dimension is collapsed by summation to produce a scalar per `(i, j)` pair per head, then softmax is applied over `j` (with padding masked to `-1e9`):

$$s_{ij} = \sum_k A'_{ijk} \in \mathbb{R}^{bs \times n_h \times n \times n}$$

$$\alpha_{ij} = \text{softmax}_j(s_{ij})$$

**6. Node aggregation**

$$h_{\text{attn},i} = \sum_j \alpha_{ij} V_j \in \mathbb{R}^{bs \times n \times d}$$

**7. Edge aggregation**

All `(nh, dk)` dimensions of $A'_{ij}$ are concatenated back to form a `d`-dim edge representation:

$$e_{\text{attn},ij} = \text{concat}_{h}(A'_{ij}) \in \mathbb{R}^{bs \times n \times n \times d}$$

**8. Global FiLM on nodes**

`y` is projected via `Linear(dy, 2d)` to produce shift and scale:

$$yN_1, yN_2 \in \mathbb{R}^{bs \times 1 \times d}$$

$$h_{\text{final},i} = yN_{1} + (1 + yN_{2}) \odot h_{\text{attn},i}$$

**9. Global FiLM on edges**

$$yE_1, yE_2 \in \mathbb{R}^{bs \times 1 \times 1 \times d}$$

$$e_{\text{final},ij} = yE_{1} + (1 + yE_{2}) \odot e_{\text{attn},ij}$$

Then in `forward`, residual connections and layer norm are applied:

$$h \leftarrow \text{LayerNorm}(h + W_N \cdot h_{\text{final}})$$

$$e \leftarrow \text{LayerNorm}(e + W_E \cdot e_{\text{final}})$$

where $W_N: \mathbb{R}^{d \to d}$ and $W_E: \mathbb{R}^{d \to d_e}$ project back to node/edge dims.

---

### `MultiHeadEdgeNode`

Same attention mechanism without the `y` conditioning pathway. Constructor takes `(d, de, n_heads)`.

### `AttentionLayer`

Full transformer layer wrapping `MultiHeadEdgeNodeWithY` followed by `FfnNodeEdge`. After the FFN, edge matrices are symmetrized: `e = 0.5 * (e + eᵀ)`.

```python
layer = AttentionLayer(d=256, de=64, dy=64, n_heads=8, y_update=True)
h, e, y, mask = layer(h, e, y, mask)
```

### `AttentionLayerWithoutY`

Variant of `AttentionLayer` for use without global conditioning. Constructor takes `(d, de, n_heads)`.

---

## `ffn.py` — Feed-Forward Networks

### `Ffn`

Position-wise FFN: `Linear(d, 2d) → ReLU → Dropout → Linear(2d, d)`. The hidden dim is always `2*d` (hardcoded).

```python
ffn = Ffn(d=256, dropout=0.1)
out = ffn(x)   # (bs, ..., 256)
```

### `FfnNodeEdge`

Applies two independent `Ffn` modules — one to node features, one to edge features — with residual connections and layer norm. Returns `(h, e, mask)`.

```python
ffn = FfnNodeEdge(d=256, de=64, dropout=0.1)
h_out, e_out, mask = ffn(h, e, mask)
```

---

## `mlp.py` — MLP Modules

### `MLP`

Single hidden layer MLP: `Linear(d, d_hidden) → SiLU → Linear(d_hidden, d_out)`. No LayerNorm, no dropout.

```python
mlp = MLP(d=256, d_hidden=512, d_out=128)
out = mlp(x)
```

### `MLPNodeEdge`

Output head used at the end of `GnnNodeEdges`. Applies two independent `MLP` modules to node and edge features, then symmetrizes the edge output. Hidden dims are hardcoded to `2*d` and `2*de`.

```python
mlp = MLPNodeEdge(d=256, de=64, natoms=17, nedges=5)
h_out, e_out, y, mask = mlp(h, e, y, mask)
# h_out: (bs, n, natoms)   — logits over atom types
# e_out: (bs, n, n, nedges) — logits over bond types (symmetrized)
```

### `MLPNodeEdgeWithoutY`

Same as `MLPNodeEdge` but `forward` takes `(h, e, mask)` and returns `(h, e, mask)` — no `y` passthrough.

```python
mlp = MLPNodeEdgeWithoutY(d=256, de=64, natoms=17, nedges=5)
h_out, e_out, mask = mlp(h, e, mask)
```
