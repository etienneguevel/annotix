import argparse
import os

import torch
from omegaconf import OmegaConf
from rdkit import RDLogger

from annotix_ml import BASE_DIR
from annotix_ml.graphtransf.arch.digress_meta_arch import DigressMetaArch
from annotix_ml.data.loaders import make_datasets
from annotix_ml.data.data_utils import batch_graph_to_smiles

logger = RDLogger.logger()
logger.setLevel(RDLogger.CRITICAL)


def get_args():
    parser = argparse.ArgumentParser(
        description="Generate molecules using a trained model."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        required=True,
        help="Number of molecules to generate.",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="annotix_ml",
        choices=["annotix_ml", "digress"],
        help="Mode to run the model in.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for generation."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the generated SMILES.",
    )
    return parser.parse_args()


def main():
    args = get_args()

    # Load configuration
    if not os.path.exists(args.config_path):
        raise ValueError(f"Config file not found: {args.config_path}")

    # Load default config
    default_cfg_path = BASE_DIR / "configs" / "default_config.yaml"
    default_cfg = OmegaConf.load(default_cfg_path)

    # Load user config
    user_cfg = OmegaConf.load(args.config_path)

    # Merge configs
    cfg = OmegaConf.merge(default_cfg, user_cfg)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset to get valid_elements and distributions
    # We only need the training dataset for initialization parameters
    print("Loading dataset...")
    train_dataset, _ = make_datasets(
        cfg.dataset.data_path,
        cfg.dataset.smile_column,
        cfg.dataset.split_column,
    )

    # Load model
    print(f"Loading model from {args.model_checkpoint}...")
    if args.mode == "annotix_ml":
        model = DigressMetaArch.load_pretrained(
            cfg,
            device,
            args.model_checkpoint,
            train_dataset.valid_elements,
            train_dataset.nodes_distribution,
            train_dataset.edges_distribution,
        )

    elif args.mode == "digress":
        model = DigressMetaArch.init_from_cfg(
            cfg,
            device,
            train_dataset.valid_elements,
            train_dataset.nodes_distribution,
            train_dataset.edges_distribution,
        )
        digress_model = torch.load(
            args.model_checkpoint, map_location=device, weights_only=False
        )
        if model.diffuser.state_dict().keys() != digress_model.state_dict().keys():
            print("WARNING: Model keys do not match. Model may not be as expected.")

        if [p.numel() for p in model.diffuser.parameters()] != [
            p.numel() for p in digress_model.parameters()
        ]:
            print(
                "WARNING: Model parameters do not match. Model may not be as expected."
            )

        model.diffuser = digress_model

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    model.diffuser.eval()

    print("Inferring max_nodes from dataset (checking first 100 samples)...")
    max_n = 0
    for i in range(min(100, len(train_dataset))):
        N, _ = train_dataset[i]
        max_n = max(max_n, N.shape[0])
    print(f"Inferred max_nodes (approx): {max_n}")
    # Add a buffer just in caseimport os

    max_n = int(max_n * 1.2)
    print(f"Using max_nodes: {max_n}")

    # Generate
    print(f"Generating {args.num_samples} molecules...")
    generated_smiles = []

    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size

    with torch.no_grad():
        for i in range(num_batches):
            current_batch_size = min(
                args.batch_size, args.num_samples - len(generated_smiles)
            )

            gen_N, gen_E, gen_mask = model.generate(
                num_samples=current_batch_size,
                max_nodes=max_n,
                progress_bar=True,
            )

            batch_smiles = batch_graph_to_smiles(
                gen_N, gen_E, gen_mask, model.valid_elements
            )
            generated_smiles.extend(batch_smiles)

    # Filter None values (invalid graphs)
    valid_smiles = [s for s in generated_smiles if s is not None]
    print(
        f"Generated {len(valid_smiles)} valid SMILES out of {args.num_samples} samples."
    )

    # Output
    if args.output_path:
        with open(args.output_path, "w") as f:
            for s in valid_smiles:
                f.write(f"{s}\n")
        print(f"Saved generated SMILES to {args.output_path}")

    else:
        print("Generated SMILES:")
        for s in generated_smiles:
            print(s)


if __name__ == "__main__":
    main()
