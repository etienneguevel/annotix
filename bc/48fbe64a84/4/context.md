# Session Context

## User Prompts

### Prompt 1

Implement the following plan:

# Plan: DDP training loop fixes — loss sync, unused-parameter error, and race conditions

## Context

After the DDP wiring (wrapping `self.diffuser` with DDP in `_setup_distributed`) was put in place,
two sets of problems remain:

1. **Explicit request** — the logged training loss is rank-local (each rank trains on a different
   data shard). Only rank 0 logs to wandb, so the reported loss reflects only its shard, not the
   true global batch average.

2. **Cra...

