"""
Test script: Similarity for two spectra using cosine similarity.
"""

from argparse import ArgumentParser

from loguru import logger
import pandas as pd
import torch

from annotix_ml.spectrum import Spectrum
from annotix_ml.word2vec import word_similarity, spectrum_similarity


def arguments():
    """
    Parse command line arguments.
    
    Returns:
        Namespace: Parsed arguments.
    """
    parser = ArgumentParser(description="Compute similarity between spectra.")
    parser.add_argument("--model_path", type=str, default="./model.pt", help="Path to the trained model.")
    return parser.parse_args()

if __name__ == "__main__":

    args = arguments()
    
    logger.info("Loading model from {}", args.model_path)
    model = torch.load(args.model_path, map_location=torch.device("cpu"))

    logger.info("Load test data and get documents")
    test = pd.read_csv("main/test_data/test-data.csv")
    spectra = [Spectrum(row) for _, row in test.iterrows()]
    spec_1 = spectra[0]
    spec_2 = spectra[1]

    logger.info("Similarity between two words in the model")
    doc = spec_1.document()
    word1 = doc.words[0]
    word2 = doc.words[1]

    sim1 = word_similarity(model, word1, word2)
    sim2 = word_similarity(model, word1, word1)

    logger.success(f"Similarity between {word1} and {word2}: {round(sim1, 2)}")
    logger.success(f"Similarity between {word1} and {word1}: {round(sim2, 2)}")

    logger.info("Similarity between two spectra (already processed as documents)")
    sim1 = spectrum_similarity(model, spec_1, spec_2)
    logger.success(f"Similarity between {spec_1.name} and {spec_2.name}: {round(sim1, 2)}")

    sim2 = spectrum_similarity(model, spec_1, spec_1)
    logger.success(f"Similarity between {spec_1.name} and {spec_1.name}: {round(sim2, 2)}")
