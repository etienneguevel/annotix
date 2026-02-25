# Session Context

## User Prompts

### Prompt 1

I want to refacto the repo to merge the very similar spec2encoder and graphtransf logics. The new package organization would look like :
annotix_ml
|-data (objects of graphtransf and spec2encoder merged)
|-distributed
|-graphtransf (containing both logics from graphtransf and spec2encoder)
  |-arch
  |-layers
  |-math
  |-models
|-spectraencoder
|-train

See how the different files can be merged and the import changes it would induce

### Prompt 2

[Request interrupted by user for tool use]

