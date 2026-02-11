import torch
from torch.distributed.pipelining import pipeline, SplitPoint

from annotix_ml.distributed import get_global_rank, get_global_size


def auto_model_split(model, example_input_kwargs):
    num_stages = get_global_size()
    stage_id = get_global_rank()

    num_layers = len(model.layers)

    # Calculate split points to divide num_layers into num_stages blocks
    split_indices = []
    for i in range(1, num_stages):
        idx = (i * num_layers) // num_stages
        if 0 < idx < num_layers:
            split_indices.append(idx)

    split_dict = {f"layers.{idx}": SplitPoint.BEGINNING for idx in split_indices}

    print("INIT : Distributed Pipeline Parallelism")
    print(
        *(f"Shape of input {k}: {inp.shape}" for k, inp in example_input_kwargs.items())
    )

    pipe = pipeline(
        module=model,
        mb_kwargs=example_input_kwargs,
        split_spec=split_dict,
    )

    stage = pipe.build_stage(stage_id, device=torch.device("cuda", stage_id))
    return stage
