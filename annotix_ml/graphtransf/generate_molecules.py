import argparse
import json
import os

import torch
from omegaconf import OmegaConf
from rdkit import RDLogger

from annotix_ml import BASE_DIR
from annotix_ml.graphtransf.arch.digress_meta_arch import DigressMetaArch
from annotix_ml.data.data_utils import batch_graph_to_smiles

logger = RDLogger.logger()
logger.setLevel(RDLogger.CRITICAL)


def get_args():
    parser = argparse.ArgumentParser(
        description="Generate molecules using a trained model from a save directory."
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Path to the training save directory (contains config.yaml, dataset_meta.json, and checkpoints).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        required=True,
        help="Number of molecules to generate.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for generation."
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=40,
        help="Max number of heavy atoms of generated mols.",
    )
    parser.add_argument(
        "--min-nodes",
        type=int,
        default=9,
        help="Min number of heavy atoms of generated mols.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save the generated SMILES.",
    )
    return parser.parse_args()


def resolve_checkpoint(save_dir: str) -> str:
    """Return the path to the last checkpoint in save_dir."""
    final = os.path.join(save_dir, "final.pt")
    if os.path.isfile(final):
        return final

    pt_files = [
        f
        for f in os.listdir(save_dir)
        if (f.endswith(".pt") and f.split(".")[0].isdigit())
    ]
    if pt_files:
        last_checkpoint = sorted(pt_files, key=lambda y: int(y.split(".")[0]))[-1]
        checkpoint_path = os.path.join(save_dir, last_checkpoint)

        return checkpoint_path

    raise FileNotFoundError(f"No checkpoint (.pt) found in {save_dir}")


def main():
    args = get_args()

    if not os.path.isdir(args.save_dir):
        raise ValueError(f"Save directory not found: {args.save_dir}")

    # Load config
    default_cfg_path = BASE_DIR / "configs" / "default_config.yaml"
    default_cfg = OmegaConf.load(default_cfg_path)

    user_cfg_path = os.path.join(args.save_dir, "config.yaml")
    if not os.path.isfile(user_cfg_path):
        raise FileNotFoundError(f"No config file found in {args.save_dir}")

    user_cfg = OmegaConf.load(user_cfg_path)
    cfg = OmegaConf.merge(default_cfg, user_cfg)

    # Load dataset metadata (no full dataset loading needed)
    meta_path = os.path.join(args.save_dir, "dataset_meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"dataset_meta.json not found in {args.save_dir}")

    with open(meta_path) as f:
        meta = json.load(f)

    valid_elements = meta["valid_elements"]
    nodes_distribution = torch.tensor(meta["nodes_distribution"])
    edges_distribution = torch.tensor(meta["edges_distribution"])
    max_weight = meta.get("max_weight")

    print(
        f"Generating molecules with the following atoms : {', '.join(valid_elements)}"
    )
    print("-" * 50)

    # Device
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")

    # Resolve and load checkpoint
    checkpoint_path = resolve_checkpoint(args.save_dir)
    print(f"Loading model from {checkpoint_path}...")
    model = DigressMetaArch.load_pretrained(
        cfg,
        device,
        checkpoint_path,
        valid_elements,
        nodes_distribution,
        edges_distribution,
        max_weight,
    )
    model.diffuser.eval()

    all_smiles = []

    # Extract the min-max sizes of mols to generate
    min_nodes = int(args.min_nodes)
    max_nodes = int(args.max_nodes)
    print(f"Generating molecules from {min_nodes} to {max_nodes} heavy atoms.")
    print("-" * 50)

    with torch.no_grad():
        samples_generated = 0
        while samples_generated < args.num_samples:
            current_bs = min(args.batch_size, args.num_samples - samples_generated)

            gen_N, gen_E, gen_mask = model.generate(
                num_samples=current_bs,
                max_nodes=max_nodes,
                min_nodes=min_nodes,
                progress_bar=True,
            )
            all_smiles.extend(
                batch_graph_to_smiles(gen_N, gen_E, gen_mask, model.valid_elements)
            )
            samples_generated += current_bs

    all_smiles = all_smiles[: args.num_samples]
    valid_smiles = [s for s in all_smiles if s is not None]
    print(
        f"Generated {len(valid_smiles)} valid SMILES out of {len(all_smiles)} samples "
        f"(validity: {len(valid_smiles) / max(len(all_smiles), 1):.1%})."
    )

    if args.output_path:
        with open(args.output_path, "w") as f:
            for s in valid_smiles:
                f.write(f"{s}\n")
        print(f"Saved generated SMILES to {args.output_path}")
    else:
        print("Generated SMILES:")
        for s in valid_smiles:
            print(s)


if __name__ == "__main__":
    main()
