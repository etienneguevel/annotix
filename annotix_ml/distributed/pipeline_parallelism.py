import torch
import torch.distributed.pipelining as pp

from annotix_ml.distributed import get_global_rank, get_global_size


def iterative_model_split(model):
    num_stages = get_global_size()
    stage_id = get_global_rank()
    device = torch.device("cuda")

    num_layers = len(model.layers)
    layers_per_stage = num_layers // num_stages

    start_layer = stage_id * layers_per_stage
    end_layer = (stage_id + 1) * layers_per_stage

    if stage_id == num_stages - 1:
        end_layer = num_layers

    model.layers = model.layers[start_layer:end_layer]

    stage = pp.PipelineStage(model, stage_id, num_stages, device)
    return stage
