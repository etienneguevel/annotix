from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
import torch

from annotix_ml.spectrum import get_documents
from annotix_ml.word2vec import train, process_text

import argparse

parser = argparse.ArgumentParser(description="Train a word2vec model.")
parser.add_argument("--context_size", type=int, default=2, help="Size of context window")
parser.add_argument("--embedding_dim", type=int, default=10, help="Dimension of word embeddings")
parser.add_argument("--algo", type=str, choices=["ngram", "cbow"], default="ngram", help="Algorithm: 'ngram' or 'cbow'")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
parser.add_argument("--device", type=str, default="mps", help="GPU device to use (e.g., 'cpu', 'cuda', 'mps')")
parser.add_argument("--epoch", type=int, default=10, help="Number of training epochs")
args = parser.parse_args()

logger.info("Load reference data and get documents")
references = pd.read_csv("main/test_data/references.csv")

logger.info("Filter by the charge")
references = references[references.charge == "1+"]
reference_documents = get_documents(references)
logger.info(f"{len(reference_documents)} reference documents loaded")

# Extract the words from the data, a.k.a. the spectra peaks
train_data = []
for doc in reference_documents:
    train_data += doc.words

# Prepare the docuemnts for training
ngrams, vocab, word_to_ix = process_text(train_data, context_size=args.context_size, type=args.algo)
model, losses = train(ngrams=ngrams, vocab=vocab, word_to_ix=word_to_ix, embedding_dim=args.embedding_dim, context_size=args.context_size, 
                      epochs=args.epoch, device=args.device, batch_size=args.batch_size, algo=args.algo)

fig = plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
fig.savefig("losses.png")

logger.info("Saving the model in ./model.pt")
torch.save(model, "./model.pt")

logger.info("Model training complete")
