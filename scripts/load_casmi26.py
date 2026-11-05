import os

import zipfile
import kagglehub
import pandas as pd

from annotix_ml import BASE_DIR

OUT_PATH = BASE_DIR / "data/enveda-CASMI26"


def main():
    # Make the saving dirs
    os.makedirs(OUT_PATH, exist_ok=True)

    # Download latest version
    path_to_zip_file = OUT_PATH.parent / "enveda-CASMI26-molecule-id-mass-spectra.zip"
    if not path_to_zip_file.exists():
        _ = kagglehub.competition_download(
            "enveda-CASMI26-molecule-id-mass-spectra", output_dir=str(OUT_PATH)
        )

    with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
        zip_ref.extractall(OUT_PATH)

    # Load the training data
    df_train = pd.read_parquet(OUT_PATH / "train.parquet")

    # Shuffle the data and sample part of its for eval / test (in case of need)
    col = df_train.columns.get_loc("split")
    df_train = df_train.sample(frac=1, random_state=12)
    df_train["split"] = "train"
    df_train.iloc[-20000:-10000, col] = "val"
    df_train.iloc[-10000:, col] = "test"

    df_train.to_parquet(OUT_PATH / "train_val_test.parquet", index=False)


if __name__ == "__main__":
    main()
