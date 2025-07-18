"""
Train Spec2Vec model
"""

"""
Train a Spec2Vec model, i.e. Word2Vec on spectrum data
"""
import sys

from loguru import logger
from matchms import calculate_scores, Spectrum
import matchms.filtering as msfilters
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from spec2vec import SpectrumDocument, Spec2Vec
from spec2vec.model_building import train_new_word2vec_model
from tqdm import tqdm


def spectrum_processing(s):
    """This is how one would typically design a desired pre- and post-processing pipeline."""
    s = msfilters.default_filters(s)
    s = msfilters.add_parent_mass(s)
    s = msfilters.normalize_intensities(s)
    # s = msfilters.reduce_to_number_of_peaks(s, n_required=10, ratio_desired=0.5, n_max=500)
    s = msfilters.select_by_mz(s, mz_from=0, mz_to=1000)
    s = msfilters.add_losses(s, loss_mz_from=10.0, loss_mz_to=200.0)
    s = msfilters.require_minimum_number_of_peaks(s, n_required=10)
    return s


def get_peaks(peaks_list):
    if not isinstance(peaks_list, str):
        return None
    # peaks = np.array([np.array(item.split(" "), dtype=float) for item in peaks_list.split("\\n") if item and isinstance(item, str)])
    peaks = []
    for item in peaks_list.split("\\n"):
        if item:
            try:
                peaks.append(np.array(item.split(" "), dtype=float))
            except AttributeError as err:
                print("Got ya!")
                print(err)
                sys.exit(0)
    peaks = np.array(peaks)

    sorted_indexes = np.argsort(peaks[:, 0])
    return peaks[sorted_indexes]

def get_spectrum(series, verbose=True):
    peaks = get_peaks(series["peaks_list"])
    if peaks is None:
        return None

    metadata = dict(
        pepmass=series["pepmass"],
        smiles=series["smiles"],
        compound_name=series["compound_name"],
        charge=series["charge"]
    )

    if verbose:
        print(f"Reading spectrum for {metadata.get('id')}")
        print(f"Pepmass:            {metadata.get('pepmass')}")
        print(f"Smiles:             {metadata.get('smiles')}")
        print(f"Number of peaks:    {len(peaks)}")


    return Spectrum(mz=peaks[:, 0], intensities=peaks[:, 1], metadata=metadata)

def plot_spectrum(spectrum, other_spectrum=None):
    
    if not other_spectrum:
        spectrum.plot() #Plot one spectrum
    else:
        spectrum.plot_against(other_spectrum)
    plt.show()

def get_documents(input_data):
    spectrums = [get_spectrum(row, verbose=False) for _, row in tqdm(input_data.iterrows(), desc="Fetch the spectrums", total=len(input_data))]

    # print(spectrum.losses) #Empty…

    # plot_spectrum(spectrums[0])
    # plot_spectrum(spectrums[0], spectrums[12])

    spectrums = [spectrum_processing(spectrum) for spectrum in spectrums if spectrum is not None]
    spectrums = [spectrum for spectrum in spectrums if spectrum is not None]

    # plot_spectrum(spectrums[0], spectrums[12])

    return [SpectrumDocument(s, n_decimals=2) for s in spectrums]

if __name__ == "__main__":

    logger.info("Load reference data and get documents")
    references = pd.read_csv("references.csv")
    
    logger.info("Filter by the charge")
    references = references[references.charge == "1+"]
    logger.info(len(references))

    reference_documents = get_documents(references)

    logger.info("Load test data and get documents")
    test = pd.read_csv("test-data.csv")
    test_documents = get_documents(test)
    
    # Train spec2vec model
    logger.info("Train Spec2Vec model")
    model = train_new_word2vec_model(reference_documents, iterations=30, filename="references.model", workers=4, progress_logger=False)

    # Get the vectors
    vectors = model.wv.vectors

    # Compute the similarities scores
    logger.info("Compute the similarities")
    scores = calculate_scores(reference_documents, test_documents, Spec2Vec(model, allowed_missing_percentage=5.0))
    print(scores.to_array())


    # # Another test
    # spectrum_1 = Spectrum(mz=np.array([100, 150, 200.]), intensities=np.array([0.7, 0.2, 0.1]), metadata={'id': 'spectrum1'})
    # spectrum_2 = Spectrum(mz=np.array([100, 140, 190.]), intensities=np.array([0.4, 0.2, 0.1]), metadata={'id': 'spectrum2'})
    # spectrums = [spectrum_1, spectrum_2]

    # scores = calculate_scores(spectrums, spectrums, Spec2Vec(model, allowed_missing_percentage=70))

    # for (reference, query, score) in scores:
    #     print(f"Cosine score between {reference.get('id')} and {query.get('id')}" +
    #           f" is {score[0]:.2f} with {score[1]} matched peaks")
