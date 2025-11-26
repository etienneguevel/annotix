import os
import requests
import zipfile
import tarfile

import numpy as np
import pandas as pd

from annotix_ml import ROOT


URL_DATASET = "https://graphgt.mathcs.emory.edu/datasets/Chemistry/QM9.zip"


def process_qm9():
    # Download the dataset
    data_dir = os.path.join(ROOT.parent, "data")
    os.makedirs(data_dir, exist_ok=True)
    zip_filepath = os.path.join(data_dir, "QM9.zip")

    if not os.path.exists(zip_filepath):
        print(f"Starting download of QM9 dataset from {URL_DATASET}...")
        response = requests.get(URL_DATASET, stream=True)
        response.raise_for_status()
        with open(zip_filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Download complete. File saved to {zip_filepath}")
    else:
        print(f"QM9 dataset already exists at {zip_filepath}. Skipping download.")

    # Unzip the dataset
    extracted_dir = os.path.join(data_dir, "QM9")
    if not os.path.exists(extracted_dir):
        print(f"Unzipping {zip_filepath}...")
        try:
            with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
                zip_ref.extractall(data_dir)
        except zipfile.BadZipFile:
            with tarfile.open(zip_filepath, "r") as tar_ref:
                tar_ref.extractall(data_dir)
        os.remove(zip_filepath)
        print(f"Unzipping complete. Files extracted to {data_dir}")
    else:
        print(f"QM9 dataset already unzipped at {extracted_dir}. Skipping unzipping.")

    # Load the dataset
    data = np.load(os.path.join(extracted_dir, "label.npy"))
    df = pd.DataFrame(data, columns=["smiles"])
    df = df.sample(frac=1, random_state=24)

    # Split into train and test
    train_ratio = 0.8
    labels = ["train" for i in range(int(len(df) * train_ratio))] + [
        "test" for i in range(len(df) - int(len(df) * train_ratio))
    ]
    df["split"] = labels

    # Save the dataset
    df.to_csv(os.path.join(data_dir, "qm9.csv"), index=False)
    print(f"QM9 dataset saved to {os.path.join(data_dir, 'qm9.csv')}")


if __name__ == "__main__":
    process_qm9()
