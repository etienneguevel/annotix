from pathlib import Path

import pandas as pd
from datasets import load_dataset


import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force", action="store_true", help="Force re-download of datasets"
    )
    args = parser.parse_args()

    DATADIR = Path(__file__).parent.parent / "data"
    save_path = DATADIR / "moses.csv"

    if save_path.exists() and not args.force:
        print(
            f"Files already exist in {DATADIR}. Skipping download. Use --force to override."
        )
        return

    # Load the datasets
    ds = load_dataset("katielink/moses")
    df_train = ds["train"].to_pandas()
    df_test = ds["test"].to_pandas()

    # Sample 10000 elements from train for validation
    df_train = df_train.sample(frac=1, random_state=12)
    df_train.iloc[-10000:, 1] = "val"

    # Save them
    DATADIR.mkdir(parents=True, exist_ok=True)
    df = pd.concat([df_train, df_test])
    df.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()
