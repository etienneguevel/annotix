import torch
from torch.distributed.pipelining import pipeline, SplitPoint

from annotix_ml.distributed import get_global_rank, get_global_size


def iterative_model_split(model, example_input):
    num_stages = get_global_size()
    stage_id = get_global_rank()
    device = torch.device("cuda")

    num_layers = len(model.layers)
    layers_per_stage = num_layers // num_stages

    split_dict = {
        f"layers.{i}": SplitPoint.BEGINNING
        for i in range(layers_per_stage, num_layers, layers_per_stage)
    }

    pipe = pipeline(
        module=model,
        mb_args=example_input,
        split_spec=split_dict,
    )

    stage = pipe.get_stage_module(stage_id)
    return stage
