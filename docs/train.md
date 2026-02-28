# `annotix_ml/train/` — Training Pipeline

The train submodule provides the unified training loop, evaluation, configuration loading,
and experiment logging for both `graphtransf` and `spec2mol` modes.

---

## `train.py` — Training Loop

Entry point: `python annotix_ml/graphtransf/train/train.py --config <path>`

### High-level flow

```python
setup(cfg)                    → load + merge config
setup_model_mode(cfg)         → select arch, dataset, collator
distributed.enable(strategy)  → DDP or pipeline

for step in range(max_steps):
    batch = next(train_loader)
    pred_X, pred_E, loss = arch.forward_backward(batch)
    optimizer.step()
    lr_scheduler.step()

    if step % save_every == 0:
        torch.save(checkpoint, path)

    if step % eval_every == 0:
        do_eval(arch, val_loader, cfg)

```

### Optimizer & LR schedule

- **Optimizer:** AdamW
- **Schedule:** Cosine annealing with linear warmup

```yaml
train:
  lr: 1e-4
  weight_decay: 1e-5
  warmup_steps: 1000
  max_steps: 200000
  eval_every: 5000
  save_every: 10000
  batch_size: 64
```

### Checkpoint resume

If `cfg.run.checkpoint` is set, `load_pretrained` is called and the optimizer state
and step counter are restored. The `InfiniteSampler` is advanced by the saved step count
so training resumes from exactly the right data position.

### WandB integration

Metrics are logged to Weights & Biases when `cfg.run.wandb` is `true`. The run name
and project are taken from `cfg.run.name` and `cfg.run.project`.

In case of distributed training the main rank is logged (see `distributed.md` for
more information on main rank).

---

## `setup.py` — Configuration & Mode Selection

### `setup(args) -> DictConfig`

Takes the `argparse.Namespace` returned by `get_args()` and builds the final config:
1. Loads `configs/default_config.yaml` (all defaults)
2. Merges the experiment config at `args.config`
3. Applies any extra CLI arguments that were provided, overriding specific config keys

**Supported CLI arguments** (all optional except `--config`):

| CLI argument | Config key overridden |
|---|---|
| `--config` | path to the experiment YAML (required) |
| `--save-path` | `cfg.train.save_path` |
| `--extra-features` | `cfg.model.extra_features` (comma-separated list) |
| `--batch-size` | `cfg.train.batch_size` |
| `--num-train-steps` | `cfg.train.num_train_steps` |
| `--project-name` | `cfg.run.name` |
| `--num-diffusion-steps` | `cfg.model.diffusion_steps` |
| `--distributed-strat` | `cfg.train.distributed` |

```bash
python -m annotix_ml.graphtransf.train.train \
    --config configs/spec2mol/baseline.yaml \
    --batch-size 32 \
    --distributed-strat data
```

### `setup_model_mode(cfg) -> tuple[Dataset, Dataset, type, Callable]`

Branches based on whether `cfg.spectra_encoder` is present in the config:

| Condition | Arch class | Datasets | Collator |
|-----------|-----------|---------|---------|
| no `spectra_encoder` section | `DigressMetaArch` | `GraphDatasetFromSMILEs` | `collateGraph` |
| no `spectra_encoder` + pipeline distributed | `DigressMetaArch` | `GraphDatasetFromSMILEs` | `collateGraphStatic` (n_max auto-derived from dataset) |
| `spectra_encoder` present | `Spec2MolMetaArch` | `GraphSpecDataset` | `collateGraphSpec` |
| `spectra_encoder` present + pipeline distributed | — | — | raises `ValueError` (not supported) |

Returns `(train_dataset, valid_dataset, arch_class, collate_fn)`. Note that `arch` is returned
as a **class**, not an instance — instantiation happens later in `train()` via
`arch.init_from_cfg(...)` or `arch.load_pretrained(...)`.

```python
train_dataset, valid_dataset, arch, collate_fn = setup_model_mode(cfg)
```

When pipeline parallelism is active (`cfg.train.distributed == "pipeline"`), `collateGraphStatic`
is used with `n_max = len(train_dataset.num_atoms_dist)` so all batches have a fixed shape.

---

## `eval.py` — Evaluation

### `do_eval(model, eval_loader, device, expected_bs) -> dict[str, float]`

Runs reconstruction evaluation on the validation set. For each batch it:
1. Noises the graph with `model.noiser`
2. Runs `model.forward` to get predicted node and edge logits
3. Converts predictions and ground truth to SMILES and computes reconstruction metrics
4. Computes per-atom-type node accuracy and per-edge-type edge accuracy

Returns a flat `dict[str, float]` of averaged metrics across the full loader.

```python
eval_metrics = do_eval(
    model=digress,
    eval_loader=valid_loader,
    device=device,
    expected_bs=cfg.valid.batch_size,  # used to skip incomplete batches in pipeline mode
)
# eval_metrics keys: "accuracy_node_C", "accuracy_node_N", ...,
#                    "accuracy_edge_SINGLE", ..., "accuracy_edge_global", ...
```

> **Note:** `do_eval` only measures reconstruction quality. Generation metrics (validity, …)
> are computed separately by `generate_samples` / `generate_samples_from_spec`.

---

### `generate_samples(model, num_samples, num_nodes_dist, batch_size) -> dict[str, list]`

Generates molecules unconditionally by sampling node counts from `num_nodes_dist` and running
the full reverse diffusion chain in batches. Trims the output to exactly `num_samples` to avoid
overshoot bias in validity metrics.

```python
gen_metrics = generate_samples(
    model=digress,
    num_samples=1000,
    num_nodes_dist=train_dataset.num_atoms_dist,  # (max_atoms,) probability tensor
    batch_size=cfg.valid.batch_size,
)
```

**Returns** `dict[str, list]` with keys:

| Key | Description |
|-----|-------------|
| `all_gen_smiles` | SMILES strings from standard RDKit conversion (`None` if invalid) |
| `all_gen_smiles_digress` | SMILES strings using the DiGress largest-fragment method |
| `validity` | Per-sample float — `1.0` if SMILES is non-null, else `0.0` |
| `validity_digress` | Same, for the DiGress conversion |

---

### `generate_samples_from_spec(model, num_samples, eval_dataloader) -> dict[str, list]`

Generates molecules conditioned on spectra drawn from `eval_dataloader`. For each batch, node
counts are matched to the ground-truth molecules (`n = mask.sum(-1)`). Iterates until
`num_samples` have been generated; stops early if the dataloader is exhausted.

```python
gen_metrics = generate_samples_from_spec(
    model=digress,
    num_samples=1000,
    eval_dataloader=valid_loader,
)
```

**Returns** `dict[str, list]` with keys:

| Key | Description |
|-----|-------------|
| `all_gen_smiles` | Generated SMILES (standard conversion) |
| `all_gen_smiles_digress` | Generated SMILES (DiGress largest-fragment conversion) |
| `validity` | Per-sample validity flag (standard) |
| `validity_digress` | Per-sample validity flag (DiGress) |
| `true_smiles` | Ground-truth SMILES from the batch |
| `tan_sim` | Per-sample Tanimoto similarity between generated and true molecule |
| `mces` | Per-sample MCES distance between generated and true molecule |

---

### `allreduce_eval_metrics(eval_metrics: dict[str, float]) -> dict[str, float]`

Gathers `eval_metrics` dicts from all DDP ranks via `all_gather_object` and returns a single
dict where each key is averaged across all ranks that reported it.

### `allreduce_gen_metrics(gen_metrics: dict[str, list]) -> dict[str, list]`

Gathers `gen_metrics` dicts from all DDP ranks via `all_gather_object` and **concatenates**
the per-sample lists, giving a single combined dict over all ranks. Used to merge the partial
generation results when each rank generates only `num_samples // world_size` molecules.

---

## `log_utils.py` — WandB Logging

### `create_gen_samples_table(smiles_list: list[str]) -> wandb.Table`

Creates a WandB table from a list of generated SMILES strings. Each row contains:
- The SMILES string of the true and generated molecules
- A rendered 2D molecule image (PNG via RDKit) of the two molecules
- Validity flag
- Metrics (MCES and tanimoto similarity)

```python
table = create_gen_samples_table(smiles)
wandb.log({"generated_molecules": table})
```

Invalid SMILES (RDKit parse failure) are included as rows with a blank image and `valid=False`.

---

## Configuration reference

Configs follow a **two-level merge** pattern: every key not set in the experiment config is
inherited from `configs/default_config.yaml`. Experiment configs only need to specify what
differs from the defaults.

### `configs/default_config.yaml` — all defaults

```yaml
run:                          # project/run name — set in experiment config

dataset:
  smile_column: smiles        # CSV column containing SMILES strings
  split_column: split         # CSV column used for train/val split
  val_tag: test               # value in split_column that identifies validation rows
  cache_path: null            # base path for .pt cache files
  max_nodes: null             # discard molecules with more atoms than this

model:
  d: 256                      # node hidden dimension
  de: 64                      # edge hidden dimension
  dy: 64                      # global (y) hidden dimension
  n_heads: 8                  # number of attention heads
  n_layers: 5                 # number of transformer layers
  y_update: False             # whether to update y at each layer
  no_y: True                  # disable y pathway (unconditional mode)
  diffusion_steps: 500        # number of diffusion timesteps T
  noise_strategy: distribution
  num_ev: 2                   # number of Laplacian eigenvectors
  extra_features:
    - laplacian_embedding
    - node_cycle
    - valence_features

train:
  batch_size: 512
  num_train_steps: 40000
  loss_ratio: 1               # edge loss weight relative to node loss
  starting_learning_rate: 0.0002
  final_learning_rate: 0.000001
  learning_rate_schedule: cosine
  save_steps: 1000            # checkpoint every N steps
  num_workers: 4
  seed: 24
  distributed:                # null, "data" (DDP), or "pipeline" (GPipe)
  num_microbatches:           # number of micro-batches for pipeline parallelism

valid:
  batch_size: 512
  num_samples: 1024           # molecules to generate during eval
  num_eval_steps: 4000        # run eval every N training steps
```

### Example experiment config — `configs/spec2mol/spectral_feat/baseline.yaml`

Only keys that differ from the defaults need to be specified.

```yaml
run:
  project: annotix_msg
  name: baseline_ms2

dataset:
  data_path: data/msg/labels_.csv          # required — path to the CSV
  spec_folder: data/msg/spec_files         # required for spec2mol — folder of .ms files
  subform_folder: data/msg/subformulae/default_subformulae
  cache_path: data/cache/msg_50nodes
  val_tag: test
  max_nodes: 40

model:
  extra_features:
    - laplacian_embedding
    - node_cycle
    - valence_features
    - spectra_fingerprint        # adds spectral conditioning to extra features

spectra_encoder:                 # presence of this section activates Spec2MolMetaArch
  form_embedder: pos-cos
  output_size: 4096
  hidden_size: 512
  spectra_dropout: 0
  top_layers: 1
  magma_modulo: 2048
  peak_attn_layers: 2
  set_pooling: intensity
  pairwise_featurization: True
  num_heads: 8
  embed_instrument: False
  inten_transform: float
  no_diffs: False
  refine_layers: 4
  checkpoint_path: checkpoints/encoder_msg.pt  # optional pretrained encoder weights

train:
  num_train_steps: 100000
  save_path: logs/msg/baseline_no_ms2
  save_steps: 4000
  distributed: "pipeline"
  num_microbatches: 4
```
