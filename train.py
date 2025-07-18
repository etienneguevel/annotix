"""
Train Spec2Vec model on spectrum data.

This script loads reference and test spectra from CSV files, processes them,
trains a Spec2Vec model, and computes similarity scores between spectra.
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
    """
    Apply a series of preprocessing and filtering steps to a spectrum.

    Args:
        s (Spectrum): Input spectrum.

    Returns:
        Spectrum: Processed spectrum.
    """
    s = msfilters.default_filters(s)
    s = msfilters.add_parent_mass(s)
    s = msfilters.normalize_intensities(s)
    # s = msfilters.reduce_to_number_of_peaks(s, n_required=10, ratio_desired=0.5, n_max=500)
    s = msfilters.select_by_mz(s, mz_from=0, mz_to=1000)
    s = msfilters.add_losses(s, loss_mz_from=10.0, loss_mz_to=200.0)
    s = msfilters.require_minimum_number_of_peaks(s, n_required=10)
    return s


def get_peaks(peaks_list):
    """
    Parse a string of peaks into a sorted numpy array.

    Args:
        peaks_list (str): String containing peaks, separated by newlines.

    Returns:
        np.ndarray: Sorted array of peaks by m/z.
    """
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
    """
    Convert a pandas Series to a Spectrum object.

    Args:
        series (pd.Series): Row from dataframe containing spectrum info.
        verbose (bool): If True, print spectrum details.

    Returns:
        Spectrum or None: Spectrum object or None if peaks are missing.
    """
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
    """
    Plot one or two spectra.

    Args:
        spectrum (Spectrum): First spectrum to plot.
        other_spectrum (Spectrum, optional): Second spectrum to plot against.
    """
    
    if not other_spectrum:
        spectrum.plot() #Plot one spectrum
    else:
        spectrum.plot_against(other_spectrum)
    plt.show()

def get_documents(input_data):
    """
    Convert a dataframe of spectrum data to a list of SpectrumDocument objects.

    Args:
        input_data (pd.DataFrame): DataFrame containing spectrum data.

    Returns:
        list: List of SpectrumDocument objects.
    """
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
