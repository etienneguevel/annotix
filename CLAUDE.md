# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Annotix ML predicts unknown molecular structures from MS/MS spectra using graph-based deep learning. The core approach uses a DiGress diffusion model to generate molecules as graphs (atoms = nodes, bonds = edges). Requires Python 3.12.

## Documentation

Detailed submodule documentation lives in `docs/`:

- [`docs/data.md`](docs/data.md) — datasets, collators, samplers, atom/bond constants
- [`docs/graphtransf/arch.md`](docs/graphtransf/arch.md) — `DigressMetaArch`, `Spec2MolMetaArch`
- [`docs/graphtransf/models.md`](docs/graphtransf/models.md) — `GnnNodeEdges`, `NoisingModel`
- [`docs/graphtransf/layers.md`](docs/graphtransf/layers.md) — attention, embeddings, FFN, MLP
- [`docs/graphtransf/math.md`](docs/graphtransf/math.md) — losses, metrics, extra features, noise sampling
- [`docs/spectraencoder.md`](docs/spectraencoder.md) — MS2 encoder, featurizers, data objects
- [`docs/distributed.md`](docs/distributed.md) — DDP and pipeline parallel utilities
- [`docs/train.md`](docs/train.md) — training loop, eval, config setup, logging

## Common Commands

```bash
# Install (uses uv for dependency management)
pip install -e .
pip install -e ".[dev]"          # dev tools: ruff, pytest, pre-commit, pylint

# Tests
pytest tests/
pytest tests/graphtransf/models/test_gnn.py          # single file
pytest tests/graphtransf/models/test_gnn.py::test_fn  # single test

# Linting & formatting (ruff via pre-commit)
pre-commit run --all-files

# Training
python -m annotix_ml.graphtransf.train.train --config configs/experiments/baseline.yaml

# Distributed training (SLURM/torchrun)
torchrun --nproc_per_node 4 -m annotix_ml.graphtransf.train.train --config configs/...

# Molecule generation
python -m annotix_ml.graphtransf.generate_molecules --num_samples 100 --model_checkpoint logs/model.pt --config_path configs/experiments/baseline.yaml

# Data loading scripts
python scripts/load_qm9.py
python scripts/load_moses.py
python scripts/load_msg.py
```

## Architecture

### Module overview

- **`annotix_ml/data/`** — Graph datasets (`GraphDatasetFromSMILEs`, `GraphSpecDataset`), collators, `InfiniteSampler`, atom/bond constants.
- **`annotix_ml/graphtransf/`** — Primary module. DiGress diffusion model for molecule generation as graphs. Includes `arch/` (orchestrators), `models/` (GNN + noiser), `layers/`, `math/`, `train/`.
- **`annotix_ml/spectraencoder/`** — Transformer-based MS/MS spectrum encoder. Processes peaks (m/z, intensity, peak type) through `FormulaTransformer` with peak attention and set pooling. Outputs a spectral fingerprint (default 4096-d).
- **`annotix_ml/distributed/`** — SLURM-aware distributed training. Two strategies: `"data"` (DDP with gradient sync) and `"pipeline"` (GPipe with layers split across ranks via `auto_model_split()`).

### MetaArch pattern

`DigressMetaArch` and `Spec2MolMetaArch` are **orchestrator classes** (not `nn.Module`). They:
- Wrap the diffuser (`GnnNodeEdges`) + noiser (`NoisingModel`) as owned modules
- Provide class methods `init_from_cfg()` and `load_pretrained()` for initialization
- Compute extra features (Laplacian, cycles, valence) before the forward pass
- Handle distributed setup and `forward_backward()` for training
- `Spec2MolMetaArch` adds `spectra_encoder` + `merge_function` (trainable Linear projection)

### Key files

```
annotix_ml/
├── data/
│   ├── atoms_data.py          VALID_ELEMENTS, TYPE_EDGES constants
│   ├── dataset.py             GraphDatasetFromSMILEs
│   ├── spec_dataset.py        GraphSpecDataset (graph + spectra pairs)
│   ├── datacollator.py        collateGraph, collateGraphStatic, collateGraphSpec
│   ├── samplers.py            InfiniteSampler
│   ├── loaders.py             make_datasets, make_spec_datasets
│   └── data_utils.py          mask_any_tensor, graph_to_smiles helpers
├── graphtransf/
│   ├── arch/
│   │   ├── digress_meta_arch.py   DigressMetaArch orchestrator
│   │   └── spec2mol_meta_arch.py  Spec2MolMetaArch (extends above)
│   ├── models/
│   │   ├── gnn.py             GnnNodeEdges: graph neural network backbone
│   │   └── noising.py         NoisingModel: forward diffusion + posterior
│   ├── layers/                attention (FiLM-modulated), embeddings, FFN, MLP
│   ├── math/                  digress_loss, metrics, extra features, noise schedule
│   ├── train/
│   │   ├── train.py           unified training loop
│   │   ├── setup.py           config loading & setup_model_mode() branching
│   │   ├── eval.py            do_eval, generate_samples, allreduce helpers
│   │   └── log_utils.py       WandB table generation
│   └── generate_molecules.py  CLI entry point for inference
├── spectraencoder/
│   ├── model/
│   │   ├── spectra_encoder.py  SpectraEncoder, SpectraEncoderGrowing
│   │   ├── modules.py          FormulaTransformer, FPGrowingModule
│   │   └── form_embedders.py   FloatFeaturizer, IntFeaturizer, FourierFeaturizer
│   └── data/
│       ├── objects.py          Spectra, Mol data classes
│       ├── featurizers.py      PeakFormula, FingerprintFeaturizer
│       ├── splitter.py         SpectraSplitter, PresetSpectraSplitter
│       └── utils.py            parse_spectra, formula_to_dense, ION_LST
└── distributed/
    ├── __init__.py             enable, is_main_process, get_global_rank/size
    └── pipeline_parallelism.py auto_model_split, save/load_checkpoint
```

### Unified training pipeline

`train/setup.py` has `setup_model_mode(cfg)` which branches based on `cfg.run.mode`:
- **graphtransf mode**: Uses `DigressMetaArch`, `GraphDatasetFromSMILEs`, `collateGraph`
- **spec2mol mode**: Uses `Spec2MolMetaArch`, `GraphSpecDataset`, `collateGraphSpec`

Training loop in `train.py`: noise batch → `arch.forward_backward()` → loss → optimizer step → cosine LR schedule → periodic eval + checkpoint saving.

`forward_backward()` returns `(pred_X, pred_E, loss)` and calls `loss.backward()` internally.

### Configuration system

OmegaConf YAML configs with hierarchical merging: `configs/default_config.yaml` (all defaults) → experiment config (e.g. `configs/graphtransf/qm9/...`) → CLI overrides. Key sections: `run`, `dataset`, `model`, `spectra_encoder`, `train`, `valid`.

Experiment configs live under `configs/graphtransf/` and `configs/spec2mol/`.

### Graph representation

- Atoms: one-hot over `VALID_ELEMENTS` in `data/atoms_data.py` (C, N, O, S, P, F, Cl, Br, I, B, As, Si, Se, Fe, Co, K, Na)
- Bonds: one-hot over `TYPE_EDGES` (NoBond, SINGLE, DOUBLE, TRIPLE, AROMATIC)
- Variable-length graphs use `(batch, max_nodes)` binary masks for padding throughout the pipeline

### Checkpoint format

```python
{
    "checkpoint": {
        "diffuser": ...,       # GnnNodeEdges state_dict
        "noiser": ...,         # NoisingModel state_dict
        "merge_function": ...  # (Spec2Mol only)
    },
    "optimizer": {...}
}
```

### Key patterns

- Extra features (Laplacian embedding, node cycles, valence) are computed before the forward pass and concatenated to inputs; configurable via `model.extra_features` list
- `mask_any_tensor()` utility applies masks with custom fill values; used throughout for padding
- Loss: `digress_loss()` uses CrossEntropyLoss with `ignore_index=-100` for padding; total = node_loss + loss_ratio * edge_loss
- `collateGraphStatic` (fixed `n_max`) is required for pipeline parallelism; `collateGraph` (dynamic padding) is used otherwise
- Experiment tracking via Weights & Biases (wandb)
- Test utilities in `graphtransf/test_utils.py` (`create_random_start`, `create_random_inp`)
