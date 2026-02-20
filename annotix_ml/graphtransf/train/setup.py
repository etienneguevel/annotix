from omegaconf import OmegaConf

from annotix_ml import BASE_DIR


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

    return cfg
