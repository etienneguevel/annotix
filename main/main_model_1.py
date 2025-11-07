import argparse
from pathlib import Path

from matchms import set_matchms_logger_level
from matchms.importing import load_from_mgf
import pandas as pd
import time
from loguru import logger
from torchinfo import summary

from annotix_ml.mass2smiles.network import (
    Mass2SmilesModel,
    prepro_specs_train,
    direct_spectral_encoding,
)
from annotix_ml.mass2smiles.process_spectrum import (
    spectrum_processing,
    metadata_processing,
)

set_matchms_logger_level("ERROR")

parser = argparse.ArgumentParser()
parser.add_argument("-input")  # Input directory
parser.add_argument("-output")  # Output directory
args = parser.parse_args()


start_time = time.time()

logger.info("Load MGF, process spectra and convert to dataframe")
path = Path(args.input)
spectrums = list(load_from_mgf(path.as_posix()))

spectrums = spectrums[:100]  # For testing purpose, remove for real use case

spectrums = [metadata_processing(s) for s in spectrums]
spectrums = [spectrum_processing(s) for s in spectrums]
# Omit spectrums that didn't qualify for analysis
spectrums = [s for s in spectrums if s is not None]

precs = []
IDs = []
mzs = []
ints = []
loss_mzs = []
loss_ints = []

for spec in spectrums:
    IDs.append(spec.get("feature_id"))
    precs.append(spec.get("precursor_mz"))
    mzs.append(list(spec.peaks.mz))
    ints.append(list(spec.peaks.intensities))
    loss_mzs.append(list(spec.losses.mz))
    loss_ints.append(list(spec.losses.intensities))

metadata = pd.DataFrame(
    list(zip(IDs, precs, mzs, ints, loss_mzs, loss_ints)),
    columns=[
        "feature_id",
        "precursor_mz",
        "mzs",
        "intensities",
        "loss_mzs",
        "loss_intensities",
    ],
)
# metadata.to_csv(Path(args.output) / "feature_ids_dataframe.tsv", sep='\t')

logger.info("Building training data")
train = prepro_specs_train(metadata)  # OK HERE
logger.info(f"train.shape: {train.shape}")

logger.info("Direct spectral encoding")
xtrain = direct_spectral_encoding(train, max_length=501)
logger.info(f"xtrain.shape: {xtrain.shape}")

# Set embedding dimension for the model
embedding_dim = 128

model = Mass2SmilesModel(
    units=2048,
    heads=8,  # 128 is divisible by 8
    dropout=0.1,
    dense_dropout=0.1,
    filters=256,
    num_layers=5,
    embed_dim=embedding_dim,  # Embedding dimension for transformer layers
    output_dim_smiles=512,
    output_dim_fg=71,
    input_dim=2,  # m/z and intensity
)

# smiles, fg = model.forward(xtrain)

print(
    summary(
        model,
        input_size=(xtrain.shape[0], xtrain.shape[1], xtrain.shape[2]),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        device="cpu",
    )
)

# model.load_weights("/app/misunderstood-fire-207/model")
# result= model.predict(xtrain)
# np.save(f'{args.output}/result_predict.npy', result[0])
# np.save(f'{args.output}/result_predict1.npy', result[1])

logger.success("Everything was successfully done.")
