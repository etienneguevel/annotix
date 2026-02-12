import torch
import torch.distributed as dist

from torch.distributed.checkpoint import load, save
from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_state_dict,
    StateDictOptions,
)
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
        *(
            f"Shape of input {k}: {inp.shape}"
            for k, inp in example_input_kwargs.items()
        ),
        sep="\n",
    )

    pipe = pipeline(
        module=model,
        mb_args=example_input_kwargs,
        split_spec=split_dict,
    )

    stage = pipe.build_stage(stage_id, device=torch.device("cuda", stage_id))
    return stage


def save_checkpoint(model, optimizer, path):
    dist.barrier()
    model_state, optimizer_state = get_state_dict(
        model, optimizer, options=StateDictOptions(full_state_dict=True)
    )
    save(
        {"model": model_state, "optimizer": optimizer_state},
        checkpoint_id=path,  # each rank will save its own file
    )
    dist.barrier()


def load_checkpoint(model, optimizer, path):
    dist.barrier()
    model_state, optimizer_state = get_state_dict(
        model, optimizer, options=StateDictOptions(full_state_dict=True)
    )
    load(
        {"model": model_state, "optimizer": optimizer_state},
        checkpoint_id=path,  # each rank will save its own file
    )
    # necessary if model.load_state_dict() should be called
    set_state_dict(
        model,
        optimizer,
        model_state_dict=model_state,
        optim_state_dict=optimizer_state,
        options=StateDictOptions(broadcast_from_rank0=True, full_state_dict=True),
    )
    dist.barrier()
