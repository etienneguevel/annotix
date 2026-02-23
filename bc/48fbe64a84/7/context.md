# Session Context

## User Prompts

### Prompt 1

Implement the following plan:

# Plan: Sync graphtransf changes to spec2mol

## Context

Recent improvements to the `graphtransf` training pipeline (dataset caching, DDP robustness, memory-tracker removal, correct rank-gated logging) were not propagated to the parallel `spec2mol` module. Both modules share the same DiGress backbone; keeping them in sync prevents bugs and keeps maintenance overhead low.

Recent graphtransf commits driving this:
- `b554289` — dataset caching
- `06d9aa7` — remo...

