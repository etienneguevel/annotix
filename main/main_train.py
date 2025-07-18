
from loguru import logger
from matchms import calculate_scores
import pandas as pd
from spec2vec import Spec2Vec
from spec2vec.model_building import train_new_word2vec_model

from annotix_ml.train import get_documents


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
