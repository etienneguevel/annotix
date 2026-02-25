# Session Context

## User Prompts

### Prompt 1

Implement the following plan:

# Plan: Distributed Validation with DistributedSampler + allreduce

## Context

Currently, every rank processes the **entire** validation set redundantly during eval. In DDP mode with N GPUs, this wastes N-1x compute on identical work. We want to shard the validation set across ranks using `DistributedSampler`, then aggregate metrics via collective operations.

## Changes (3 files)

### 1. `annotix_ml/train/eval.py` — Add `allreduce_eval_metrics()` helper

Add a ...

