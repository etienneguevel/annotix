# Session Context

## User Prompts

### Prompt 1

Implement the following plan:

# Refactoring Plan: Merge spec2mol into graphtransf + Reorganize Package

## Context

The `spec2mol/` package is ~90% duplicated from `graphtransf/` (training loop, dataset, collator, `smilesToGraph`). Data modules are scattered across `graphtransf/data/` and `spec2mol/data/`. This refactoring centralizes data, dissolves `spec2mol/`, and lifts training to a top-level module. **`spectraencoder/` is NOT touched.**

**Target structure:**
```
annotix_ml/
├── data...

### Prompt 2

[Request interrupted by user]

### Prompt 3

continue

