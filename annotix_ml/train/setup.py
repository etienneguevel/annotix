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
    # Load default config
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
        cfg.train.batch_size = args.batch_size

    if args.num_train_steps:
        cfg.train.num_train_steps = args.num_train_steps

    if args.project_name:
        cfg.run.name = args.project_name

    if args.num_diffusion_steps:
        cfg.model.diffusion_steps = args.num_diffusion_steps

    if args.distributed_strat:
        cfg.train.distributed = args.distributed_strat

    return cfg


def setup_model_mode(cfg, device):
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

        print("Using the Spec2MolMetaArch\n")
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
            verbose=True,
            cache_path=cfg.dataset.get("cache_path"),
            save_cache=dist.is_main_process(),
            max_nodes=cfg.dataset.get("max_nodes"),
        )

        print("Using the DigressMetaArch.\n")

        if cfg.train.get("distributed") == "pipeline":
            # Derive n_max from the distribution of number of atoms
            n_max = len(train_dataset.num_atoms_dist)
            collate_fn = partial(collateGraphStatic, n_max=n_max)
            print(f"Using static shape data collator with n_max={n_max}")

        else:
            collate_fn = collateGraph

    return train_dataset, valid_dataset, collate_fn
