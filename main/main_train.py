import pickle

from loguru import logger
from matchms import calculate_scores
import matplotlib.pyplot as plt
import pandas as pd
from spec2vec import Spec2Vec
from spec2vec.model_building import train_new_word2vec_model

from annotix_ml.spectrum import get_documents
from annotix_ml.word2vec import train, process_text


logger.info("Load reference data and get documents")
references = pd.read_csv("main/test_data/references.csv")

logger.info("Filter by the charge")
references = references[references.charge == "1+"]
reference_documents = get_documents(references)
logger.info(f"{len(reference_documents)} reference documents loaded")

logger.info("Load test data and get documents")
test = pd.read_csv("main/test_data/test-data.csv")
test_documents = get_documents(test)

reference_documents = test_documents

# Several options to train the model:

# 1. using spec2vec package
# logger.info("Train Spec2Vec model")
# model = train_new_word2vec_model(test_documents, iterations=30, filename="references.model", workers=4, progress_logger=True)

# 2. using gensim directly
# from gensim.models import Word2Vec
# model = Word2Vec(test_documents, vector_size=300, window=500, min_count=1, sg=0, negative=5, workers=4, epochs=30, compute_loss=True)

# 3. using own CBOW/NGRAM implementation for GPU support
# need to preprocess the text first

algo = "cbow"  # or "ngram"

train_data = []
for doc in reference_documents:
    train_data += doc.words
ngrams, vocab, word_to_ix = process_text(train_data, context_size=2, type=algo)
model, losses = train(ngrams=ngrams, vocab=vocab, word_to_ix=word_to_ix, embedding_dim=150, context_size=500, epochs=10, device="mps", batch_size=1024, algo=algo)

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.show()

logger.info("Model training complete")

# Get the vectors
if hasattr(model, "wv"):
    vectors = model.wv.vectors
    print(f"Number of vectors: {len(vectors)}")

# # Compute the similarities scores
# logger.info("Compute the similarities")
# scores = calculate_scores(reference_documents, test_documents, Spec2Vec(model, allowed_missing_percentage=5.0))
# print(scores.to_array())

with open("./model.pkl", "wb") as f:
    pickle.dump({"model": model, "losses": losses}, f)
