# Session Context

## User Prompts

### Prompt 1

Implement the following plan:

# Plan: Advance LR Scheduler and Train Loader on Resume

## Context

The training script was updated to detect existing checkpoints and resume from them (loading model weights via `load_pretrained`). However two key components are not yet restored:

1. **LR Scheduler** — `CosineAnnealingLR` always starts from step 0, so the LR jumps back to the initial value instead of continuing the cosine curve from where training stopped.
2. **DataLoader sampler** — `Infinit...

### Prompt 2

Is there a way to sync back to the wandb logs that were generated in the previous failed run ?

### Prompt 3

yes

### Prompt 4

[Request interrupted by user]

### Prompt 5

launching the newly written train file makes it stuck after the printing at line 413. Moreover new jobs are created on main rank which are as many as the number of processes launched with torchrun

