import torch
import torch.nn as nn
from collections import defaultdict

import annotix_ml.distributed as dist


class LayerMemoryTracker:
    """Track CUDA memory usage per layer using forward hooks.

    Registers pre- and post-forward hooks on each direct child module of a model.
    At each forward pass, records the memory allocated before and after each layer,
    computing the delta. Results are tagged with the current process rank.

    Usage:
        tracker = LayerMemoryTracker(model, device)
        # ... run forward pass ...
        mem_log = tracker.get_metrics()  # dict ready for wandb.log
        tracker.reset()
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.device = device
        self.rank = dist.get_global_rank()
        self._metrics: dict[str, list[float]] = defaultdict(list)
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._pre_mem: dict[str, int] = {}

        self._register_hooks(model)

    def _register_hooks(self, model: nn.Module):
        for name, module in model.named_children():
            # Pre-forward: snapshot memory before layer
            def pre_hook(mod, inp, layer_name=name):
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                    self._pre_mem[layer_name] = torch.cuda.memory_allocated(self.device)

            # Post-forward: compute delta
            def post_hook(mod, inp, out, layer_name=name):
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                    post_mem = torch.cuda.memory_allocated(self.device)
                    pre_mem = self._pre_mem.pop(layer_name, post_mem)
                    delta_mb = (post_mem - pre_mem) / 1e6
                    self._metrics[layer_name].append(delta_mb)

            h1 = module.register_forward_pre_hook(pre_hook)
            h2 = module.register_forward_hook(post_hook)
            self._hooks.extend([h1, h2])

    def get_metrics(self) -> dict[str, float]:
        """Return a flat dict of per-layer memory deltas (MB) for the last forward pass.

        Keys are formatted as: memory/rank_{rank}/{layer_name}_delta_MB
        Also includes the total allocated and peak memory for this rank.
        """
        result = {}
        for layer_name, deltas in self._metrics.items():
            if deltas:
                result[f"memory/rank_{self.rank}/{layer_name}_delta_MB"] = deltas[-1]

        if self.device.type == "cuda":
            result[f"memory/rank_{self.rank}/total_allocated_MB"] = (
                torch.cuda.memory_allocated(self.device) / 1e6
            )
            result[f"memory/rank_{self.rank}/peak_allocated_MB"] = (
                torch.cuda.max_memory_allocated(self.device) / 1e6
            )

        return result

    def reset(self):
        """Clear recorded metrics for the next step."""
        self._metrics.clear()
        self._pre_mem.clear()

    def remove_hooks(self):
        """Remove all registered hooks from the model."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
