import argparse
from pathlib import Path

from matchms import set_matchms_logger_level
from matchms.importing import load_from_mgf
import pandas as pd
import time
from loguru import logger
from torchinfo import summary

from annotix_ml.mass2smiles.network import Mass2SmilesModelV2, prepro_specs_train, learned_spectral_encoding
from annotix_ml.mass2smiles.process_spectrum import spectrum_processing, metadata_processing

set_matchms_logger_level("ERROR")

parser = argparse.ArgumentParser()
parser.add_argument("-input") # Input directory
parser.add_argument("-output") # Output directory
args = parser.parse_args()


start_time = time.time()

logger.info('Load MGF, process spectra and convert to dataframe')
path = Path(args.input)
spectrums = list(load_from_mgf(path.as_posix()))

spectrums = spectrums[:100]  # For testing purpose, remove for real use case

spectrums = [metadata_processing(s) for s in spectrums]
spectrums = [spectrum_processing(s) for s in spectrums]
# Omit spectrums that didn't qualify for analysis
spectrums = [s for s in spectrums if s is not None]

precs = []
IDs = []
mzs=[]
ints=[]
loss_mzs=[]
loss_ints=[]

for spec in spectrums: 
    IDs.append(spec.get("feature_id"))
    precs.append(spec.get("precursor_mz"))
    mzs.append(list(spec.peaks.mz))
    ints.append(list(spec.peaks.intensities))
    loss_mzs.append(list(spec.losses.mz))
    loss_ints.append(list(spec.losses.intensities))

metadata = pd.DataFrame(list(zip(IDs, precs,mzs,ints,loss_mzs,loss_ints)), columns=["feature_id", "precursor_mz","mzs","intensities","loss_mzs","loss_intensities" ])
# metadata.to_csv(Path(args.output) / "feature_ids_dataframe.tsv", sep='\t')

logger.info('Building training data')
train = prepro_specs_train(metadata) #OK HERE

logger.info('Learned spectral encoding (Option 2)')
mz_data, intensity_data = learned_spectral_encoding(train, max_length=501)
logger.info(f"train.shape: {train.shape}")
logger.info(f"mz_data.shape: {mz_data.shape}")
logger.info(f"intensity_data.shape: {intensity_data.shape}")

# Set embedding dimension for the model
embedding_dim = 128

model = Mass2SmilesModelV2(
    units=2048,
    heads=8,  # 128 is divisible by 8
    dropout=.1,
    dense_dropout=.1,
    filters=256,
    num_layers=5,
    embed_dim=embedding_dim,  # Embedding dimension for transformer layers
    output_dim_smiles=512,
    output_dim_fg=71,
    num_mz_bins=20000,  # Number of m/z bins for embedding
    mz_max=2000  # Maximum m/z value expected
)

# Test forward pass
smiles_output, fg_output = model.forward(mz_data, intensity_data)
logger.info(f"smiles_output.shape: {smiles_output.shape}")
logger.info(f"fg_output.shape: {fg_output.shape}")

print(summary(model, input_data=[mz_data, intensity_data], col_names=["input_size", "output_size", "num_params", "trainable"], device="cpu"))

logger.success("Everything was successfully done.")