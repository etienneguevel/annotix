"""
Similarity for two spectra using cosine similarity.
"""

from loguru import logger
import numpy as np
import pandas as pd
import torch

from annotix_ml.spectrum import get_documents
from annotix_ml.word2vec import word_similarity, embedding, spectrum_simularity



if __name__ == "__main__":

    logger.info("Load test data and get documents")
    test = pd.read_csv("main/test_data/test-data.csv")
    test_documents = get_documents(test)

    doc = test_documents[0]
    word1 = doc.words[0]
    word2 = doc.words[1]

    model = torch.load("model.pt", map_location=torch.device("cpu"))

    logger.warning("Similarity between two words in the model")
    vector = embedding(model, word1)
    print(vector)
    
    sim1 = word_similarity(model, word1, word2)
    sim2 = word_similarity(model, word1, word1)

    logger.info("Vector:", vector)
    logger.info(f"Similarity between {word1} and {word2}: {round(sim1, 2)}")
    logger.info(f"Similarity between {word1} and {word1}: {round(sim2, 2)}")

    spec_doc_1 = test_documents[0]
    spec_doc_2 = test_documents[1]

    logger.warning("Similarity between two spectra (already processed as documents)")
    sim1 = spectrum_simularity(model, spec_doc_1, spec_doc_2)
    logger.info(f"Similarity between spec1 and spec2: {round(sim1, 2)}")

    sim2 = spectrum_simularity(model, spec_doc_1, spec_doc_1)
    logger.info(f"Similarity between spec1 and spec1: {round(sim2, 2)}")


