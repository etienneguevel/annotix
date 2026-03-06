# Annotix ML — Documentation

Annotix ML predicts unknown molecular structures from MS/MS spectra using graph-based deep learning.
The core approach uses a **DiGress diffusion model** that generates molecules as graphs (atoms = nodes, bonds = edges),
optionally conditioned on mass-spectrometry fingerprints via a built-in `fp_projector`.

## Submodule Index

| Doc | Submodule | Description |
|-----|-----------|-------------|
| [data.md](data.md) | `annotix_ml/data/` | Graph datasets, collators, samplers, atom/bond constants |
| [graphtransf/arch.md](graphtransf/arch.md) | `annotix_ml/graphtransf/arch/` | `DigressMetaArch` orchestrator |
| [graphtransf/models.md](graphtransf/models.md) | `annotix_ml/graphtransf/models/` | `GnnNodeEdges` backbone and `NoisingModel` |
| [graphtransf/layers.md](graphtransf/layers.md) | `annotix_ml/graphtransf/layers/` | Attention, embeddings, FFN, MLP |
| [graphtransf/math.md](graphtransf/math.md) | `annotix_ml/graphtransf/math/` | Losses, metrics, extra features, noise sampling |
| [spectraencoder.md](spectraencoder.md) | `annotix_ml/spectraencoder/` | MS2 spectra encoder, featurizers, data objects |
| [distributed.md](distributed.md) | `annotix_ml/distributed/` | DDP and pipeline-parallel distributed training |
| [train.md](train.md) | `annotix_ml/train/` | Training loop, evaluation, config setup, logging |

## High-level Architecture

```
MS/MS spectrum (optional)
      │
      ▼
 SpectraEncoder          (FormulaTransformer → fingerprint [bs, 4096])
      │ fp_projector     (Linear or MLP → [bs, morgan_nbits])
      ▼
 DigressMetaArch
  ├── NoisingModel       (adds forward-diffusion noise to graph)
  └── GnnNodeEdges       (denoises graph at each step, conditioned on y)
      │
      ▼
  Generated molecule graph  (atoms + bonds)
```

## Graph Representation

All molecules are represented as:

- **Nodes** `X`: `(bs, n, natoms)` float32 one-hot tensor over `VALID_ELEMENTS`
- **Edges** `E`: `(bs, n, n, nedges)` float32 one-hot tensor over `TYPE_EDGES`
- **Mask** `mask`: `(bs, n)` bool tensor — `True` for real atoms, `False` for padding
- **Global** `y`: `(bs, d)` float32 conditioning vector (e.g., spectral fingerprint)

Elements: C, N, O, S, P, F, Cl, Br, I, B, As, Si, Se, Fe, Co, K, Na
Bond types: NoBond, SINGLE, DOUBLE, TRIPLE, AROMATIC

## Checkpoint Format

```python
{
    "checkpoint": {
        "model":        ...,  # GnnNodeEdges state_dict
        "fp_projector": ...,  # (only when fp_projector is active) Linear or MLP state_dict
    },
    "optimizer": { ... }
}
```

## Quick Start

```bash
# Training
python annotix_ml/graphtransf/train/train.py \
    --config configs/graphtransf/qm9/medium_arhc/long.yaml

# Molecule generation
python -m annotix_ml/graphtransf/generate_molecules.py \
    --num_samples 100 \
    --model_checkpoint logs/model.pt \
    --config_path configs/graphtransf/qm9/medium_arhc/long.yaml
```

### Distributed Training with torchrun

Use `torchrun` to launch data-parallel (DDP) training across multiple GPUs.
Set `model.distributed_strategy: "data"` in your config.

**Single node, multiple GPUs:**

```bash
torchrun \
    --standalone \
    --nproc_per_node 4 \
    -m annotix_ml/graphtransf/train/train.py \
    --config configs/experiments/baseline.yaml \
    model.distributed_strategy=data
```

**Multiple nodes** (e.g. 2 nodes × 4 GPUs = 8 processes):

```bash
# Run on every node — set MASTER_ADDR to the hostname of node 0
torchrun \
    --nnodes 2 \
    --nproc_per_node 4 \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port 29500 \
    -m annotix_ml/graphtransf/train/train.py \
    --config configs/experiments/baseline.yaml \
    model.distributed_strategy=data
```

**Pipeline parallelism** (layers split across GPUs, requires static shapes):
> ⚠️ pp has been deprecated since 02/2026 and should not work with most of the
> current objects.
> PP raise issues a little bit everywhere in the implementation, but the most
> important one is with the generation of new samples.
```bash
torchrun \
    --standalone \
    --nproc_per_node 4 \
    -m annotix_ml.graphtransf.train.train \
    --config configs/experiments/baseline.yaml \
    model.distributed_strategy=pipeline
```

> **Note:** Pipeline parallelism requires a fixed maximum graph size. Set `dataset.max_nodes`
> in your config — the collator will pad all graphs to that size.

See [`distributed.md`](distributed.md) for implementation details.
