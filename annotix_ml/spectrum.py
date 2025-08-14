"""
Train Spec2Vec model on spectrum data.

This script loads reference and test spectra from CSV files, processes them,
trains a Spec2Vec model, and computes similarity scores between spectra.
"""

import sys

import matchms
import matchms.filtering as msfilters
import matplotlib.pyplot as plt
import numpy as np
from spec2vec import SpectrumDocument
from tqdm import tqdm


class Spectrum(matchms.Spectrum):
    def __init__(self, input_data):
        mz, intensities, metadata = get_spectrum(input_data)
        super().__init__(mz=mz, intensities=intensities, metadata=metadata)

    def document(self, n_decimals=2, processing_first=True):
        """
        Convert the Spectrum to a SpectrumDocument with specified decimal precision.

        Args:
            n_decimals (int): Number of decimal places for m/z and intensity values.

        Returns:
            SpectrumDocument: Document representation of the spectrum.
        """

        spectrum = spectrum_processing(self) if processing_first else self
        return SpectrumDocument(spectrum, n_decimals=n_decimals)

    def __repr__(self):
        string = f"""
Spectrum(id={self.metadata.get('id')}, mz=[{self.mz[0]}, ..., {self.mz[-1]}], intensities=[{self.intensities[0]}, ..., {self.intensities[-1]}])

Metadata:
Spectrum ID:     {self.metadata.get('id')}
Smiles:          {self.metadata.get('smiles')}
Compound name:   {self.metadata.get('compound_name')}
Pepmass:         {self.metadata.get('pepmass')}
Charge:          {self.metadata.get('charge')}
Number of peaks: {len(self.peaks)}
"""
        return string.strip()

    def __str__(self):
        return self.__repr__()

    @property
    def name(self):
        """
        Return the compound name from the metadata.
        """
        return self.metadata.get("compound_name", "Unknown Compound")
    @property
    def smiles(self):
        """
        Return the SMILES representation from the metadata.
        """
        return self.metadata.get("smiles", "Unknown SMILES")

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

def get_spectrum(series):
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
        charge=series["charge"],
        id=series["spectral_data_id"]
    )

    return peaks[:, 0], peaks[:, 1], metadata

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
