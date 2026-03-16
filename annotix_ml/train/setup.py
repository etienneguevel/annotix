import os
from functools import partial

import torch
from omegaconf import OmegaConf

import annotix_ml.distributed as dist
from annotix_ml import BASE_DIR
from annotix_ml.data.datacollator import (
    collateGraph,
    collateGraphStatic,
)
from annotix_ml.data.loaders import make_datasets, make_spec_datasets
from annotix_ml.spectraencoder.model.spectra_encoder import SpectraEncoderGrowing


def setup(args):
    # Check if there is a pretrain path indicated, if so load cfg from it as base
    if args.pretrain_path is not None:
        default_cfg = OmegaConf.load(os.path.join(args.pretrain_path, "config.yaml"))
        default_cfg.train.pretrain_path = args.pretrain_path

    else:
        # Else load default config
        default_cfg_path = BASE_DIR / "configs" / "default_config.yaml"
        default_cfg = OmegaConf.load(default_cfg_path)

    # Load user config
    user_cfg = OmegaConf.load(args.config)

    # Merge configs
    cfg = OmegaConf.merge(default_cfg, user_cfg)

    # Replace the other args indicated in the command line
    if args.save_path:
        cfg.train.save_path = args.save_path

    if args.extra_features:
        cfg.model.extra_features = [i.strip() for i in args.extra_features.split(",")]

    if args.batch_size:
        cfg.train.mini_batch_size = args.batch_size

    if args.num_train_steps:
        cfg.train.num_train_steps = args.num_train_steps

    if args.project_name:
        cfg.run.name = args.project_name

    if args.num_diffusion_steps:
        cfg.model.diffusion_steps = args.num_diffusion_steps

    if args.distributed_strat:
        cfg.train.distributed = args.distributed_strat

    if args.main_rank_only_generation:
        cfg.valid.main_rank_only_generation = True

    return cfg


def setup_model_mode(cfg, device):
    """
    Build train/validation datasets and select the appropriate collate function based on config.

    Branches on whether a `spectra_encoder` section is present in `cfg`:
    - **Spec2Mol path** (`cfg.spectra_encoder` present): Loads and freezes a
      `SpectraEncoderGrowing` model, then builds `GraphSpecDataset` instances via
      `make_spec_datasets`. The encoder is passed into the dataset so fingerprints are
      pre-computed at dataset time. Uses `collateGraph` as the collate function.
    - **Graph-only path** (`cfg.spectra_encoder` absent): Uses `GraphDatasetWithFP` when
      `"spectra_fingerprint"` is listed in `cfg.model.extra_features`, otherwise uses
      `GraphDatasetFromSMILEs`. For pipeline-parallel training, switches to
      `collateGraphStatic` with a fixed `n_max` derived from the training set.

    Args:
        cfg (DictConfig): Merged OmegaConf configuration object.
        device (torch.device): Device on which the spectra encoder (if any) is placed.

    Returns:
        tuple[Dataset, Dataset, Callable]: A tuple (train_dataset, valid_dataset, collate_fn).
    """
    if cfg.get("spectra_encoder") is not None:
        spectra_encoder = SpectraEncoderGrowing.init_from_cfg(cfg)
        if (ckpt_path := cfg.spectra_encoder.get("checkpoint_path")) is not None:
            model_dict = torch.load(BASE_DIR / ckpt_path)
            spectra_encoder.load_state_dict(model_dict)

        spectra_encoder = spectra_encoder.to(device)

        print("Building graph-spec datasets.")
        train_dataset, valid_dataset = make_spec_datasets(
            cfg,
            spectra_encoder,
            device,
            save_cache=cfg.dataset.get("cache_path"),
            verbose=dist.is_main_process(),
        )

        collate_fn = collateGraph

    else:
        if "spectra_fingerprint" in cfg.model.get("extra_features"):
            fingerprint = True
            print("Building datasets pre-computing fingerprints from graphs.")

        else:
            fingerprint = False
            print("Building classic Graph datasets.")

        train_dataset, valid_dataset = make_datasets(
            cfg.dataset.data_path,
            cfg.dataset.smile_column,
            cfg.dataset.split_column,
            cfg.dataset.val_tag,
            fingerprint=fingerprint,
            verbose=dist.is_main_process(),
            cache_path=cfg.dataset.get("cache_path"),
            save_cache=dist.is_main_process(),
            max_nodes=cfg.dataset.get("max_nodes"),
        )

        if cfg.train.get("distributed") == "pipeline":
            # Derive n_max from the distribution of number of atoms
            n_max = len(train_dataset.num_atoms_dist)
            collate_fn = partial(collateGraphStatic, n_max=n_max)
            print(f"Using static shape data collator with n_max={n_max}")

        else:
            collate_fn = collateGraph

    return train_dataset, valid_dataset, collate_fn
