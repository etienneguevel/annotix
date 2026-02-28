# `annotix_ml/distributed/` — Distributed Training

Annotix ML supports two distributed training strategies, both managed through this submodule.

| Strategy | Config value | Description |
|----------|-------------|-------------|
| Data parallelism | `"data"` | DDP — each GPU processes a different batch slice |
| Pipeline parallelism | `"pipeline"` | GPipe — model layers split across GPUs |

---

## `__init__.py` — Distributed Environment

### `enable(...)`

Initializes the NCCL process group. Detects the launch environment automatically:
- **torchrun** — reads `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, … env vars set by torchrun (takes priority)
- **SLURM** — reads `SLURM_PROCID`, `SLURM_NTASKS`, `SLURM_JOB_NODELIST`, `SLURM_LOCALID` env vars
- **Single GPU** — falls back to a single-process group (`rank=0, world_size=1`) if CUDA is available

Raises `RuntimeError` if the environment is partially set or no GPU is found.

```python
def enable(
    *,
    set_cuda_current_device: bool = True,   # call torch.cuda.set_device(local_rank)
    overwrite: bool = False,                # allow overwriting existing env vars
    allow_nccl_timeout: bool = False,       # set NCCL_ASYNC_ERROR_HANDLING=1
    main_rank: str = "first",              # "first" (rank 0) or "last" (rank world_size-1)
)
```

After calling `enable`, `print()` is silenced on all non-main ranks.

**`main_rank`** controls which rank is considered the "main" process for logging and saving:
- `"first"` (default) — rank 0 is main. Use for DDP.
- `"last"` — the highest rank is main. Use for pipeline parallelism, where only the last stage
  produces outputs and loss.

**Actual usage from `train.py`:**

```python
import annotix_ml.distributed as dist

# main_rank is "last" for pipeline (last rank has the loss), "first" for everything else
main_rank = "last" if cfg.train.get("distributed") == "pipeline" else "first"
dist.enable(overwrite=True, main_rank=main_rank)
```

> **Note:** `enable` does not accept a strategy argument. The training strategy (`"data"` or
> `"pipeline"`) is handled separately by `arch._setup_distributed(strategy, ...)` after the
> model is built.

### Process group queries

```python
from annotix_ml import distributed

distributed.set_main_rank(rank: int) # int - choose a rank to set as main (first for ddp, last for pp)
distributed.is_main_process()        # bool — True only on main rank (if one GPU -> true)
distributed.get_global_rank()        # int — current process rank (0 to world_size-1)
distributed.get_global_size()        # int — total number of processes (world size)
distributed.get_local_rank()         # int — rank within the current node
```

These functions are safe to call before `enable()` — they return rank=0, size=1 by default.

### `_TorchDistributedEnvironment`

Internal dataclass that holds rank/size/master_addr/master_port once detected. Not intended for direct use.

---

## `DigressMetaArch._setup_distributed`

Applies the chosen distributed strategy to the arch **after** `dist.enable()` has been called
and the model has been initialised. Must be called before the training loop starts.

```python
arch._setup_distributed(
    mode,             # "data" or "pipeline"
    num_microbatches, # int — number of micro-batches per pipeline step
    example_batch,    # one real batch used to trace the pipeline graph
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `mode` | `str` | `"data"` for DDP, `"pipeline"` for GPipe. `"tensor"` is accepted but is currently a no-op. |
| `num_microbatches` | `int` | Number of micro-batches the full batch is split into for pipeline scheduling. Only used in `"pipeline"` mode. |
| `example_batch` | `tuple[Tensor, ...]` | One real training batch (nodes, edges, mask). Used to compute an example input for tracing the pipeline stages. |

### DDP mode (`"data"`)

Wraps `self.diffuser` in `torch.nn.parallel.DistributedDataParallel` with
`device_ids=[get_local_rank()]` and stores the result as `self.train_model`. Gradient
synchronisation for `diffuser` is then handled automatically by DDP.

`Spec2MolMetaArch` has extra trainable parameters (`merge_function`, `spectra_encoder`) that
live outside DDP. After each backward pass, `arch.sync_extra_gradients()` must be called to
manually all-reduce their gradients:

```python
# In train.py, after loss.backward():
if cfg.train.get("distributed") == "data":
    arch.sync_extra_gradients()
```

For `DigressMetaArch` this is a no-op. For `Spec2MolMetaArch` it all-reduces the
`merge_function` parameter gradients with `ReduceOp.AVG`.

### Pipeline mode (`"pipeline"`)

1. Slices `example_batch` down to one micro-batch (`batch_size // num_microbatches`) and runs
   `compute_extra_features` to build a fully-featurised example input.
2. Passes that example input to `auto_model_split(self.diffuser, example_input)` which traces
   and splits `diffuser` into one `PipelineStage` per GPU.
3. Wraps the stages in `ScheduleGPipe` with a custom `loss_fn` (for training) and without one
   (for eval), stored as `self.train_schedule` and `self.eval_schedule`.

Only the **first rank** receives the raw input; only the **last rank** computes the loss and
has non-`None` return values from `forward_backward`. This is why `main_rank="last"` must be
passed to `dist.enable()` when using pipeline parallelism.

**Actual call in `train.py`:**

```python
if cfg.train.get("distributed") is not None:
    example_batch = next(iter(train_loader))
    digress._setup_distributed(
        cfg.train.distributed,       # "data" or "pipeline"
        cfg.train.num_microbatches,
        example_batch,
    )
```

---

## `pipeline_parallelism.py` — Pipeline Parallel Utilities

### `auto_model_split(model, n_chunks)`

Automatically partitions a sequential `nn.Module` across `n_chunks` pipeline stages (one per GPU)
using PyTorch's `ScheduleGPipe`. Layers are split as evenly as possible by parameter count.

```python
from annotix_ml.distributed.pipeline_parallelism import auto_model_split

# Must be called after dist.enable()
model = auto_model_split(model, n_chunks=4)
```

The returned model wraps the original with a `PipelineStage` interface. Only the first rank
receives inputs; only the last rank produces outputs.

### `save_checkpoint(model, optimizer, path)`

Saves a pipeline-parallel checkpoint. Gathers all stage state dicts on rank 0 before writing.

```python
save_checkpoint(model, optimizer, path="logs/checkpoint.pt")
```

Only rank 0 actually writes the file.

### `load_checkpoint(model, optimizer, path)`

Loads a pipeline-parallel checkpoint. Distributes weights from rank 0 to all stages.

```python
load_checkpoint(model, optimizer, path="logs/checkpoint.pt")
```

---

## Usage patterns

### DDP training

```python
import annotix_ml.distributed as dist

dist.enable(overwrite=True, main_rank="first")

# Strategy is applied after model init via arch._setup_distributed
arch = DigressMetaArch.init_from_cfg(cfg, device, ...)
arch._setup_distributed("data", num_microbatches, example_batch)

# Only log/save on main process (rank 0)
if dist.is_main_process():
    wandb.log(metrics)
    torch.save(checkpoint, path)
```

### Pipeline parallel training

```python
import annotix_ml.distributed as dist

dist.enable(overwrite=True, main_rank="last")  # last rank has the loss

arch = DigressMetaArch.init_from_cfg(cfg, device, ...)
arch._setup_distributed("pipeline", num_microbatches, example_batch)

# Use static-shape collator (required for pipeline parallelism)
collate_fn = partial(collateGraphStatic, n_max=cfg.dataset.max_nodes)

# Use pipeline-aware checkpoint helpers
from annotix_ml.distributed.pipeline_parallelism import save_checkpoint, load_checkpoint
save_checkpoint(arch.diffuser, optimizer, path)
```

---

## Notes

- **Gradient synchronization** in DDP mode: `diffuser` gradients are synced automatically by DDP; extra parameters outside DDP (e.g. `merge_function` in `Spec2MolMetaArch`) require a manual `arch.sync_extra_gradients()` call after `loss.backward()`.
- **Pipeline parallelism** requires static tensor shapes — use `collateGraphStatic` instead of `collateGraph`.
- `InfiniteSampler` is distributed-aware: pass `start=distributed.get_global_rank()` and `step=distributed.get_global_size()`, or omit both to have the sampler auto-read them from the process group after `enable()` has been called.
