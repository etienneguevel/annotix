# Session Context

## User Prompts

### Prompt 1

Implement the following plan:

# Plan: Fill DDP TODOs and Fix Incomplete Implementations

## Context

The user added a `"data"` parallelism branch to `_setup_distributed` (wrapping `self.diffuser` with `DDP`),
and left two `# TODO` markers in `forward` and `forward_backward` for data-parallel redefinition.
The goal is to fill those TODOs and fix any other incomplete or incorrect implementations found during review.

---

## Issues Found

### 1. `forward_backward` — critical DDP bug (line 472)
...

### Prompt 2

the train loss for logging is made only on main rank, propose changes to sync the loss on each process. Check if there are other parts of the train loop that could make it fail for none main ranks

### Prompt 3

Implement these changes, further your analysis taking into account that the current code failed on the generate part with following message for each rank : RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. This error indic
ates that your module has parameters that were not used in producing loss. You can enable unused parameter detection by passi
ng the keyword argument `find_unused_parameters=True` to `torch.nn.parallel.DistributedDataParallel`,...

### Prompt 4

[Request interrupted by user for tool use]

