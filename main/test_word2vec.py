"""
Main script for training and plotting the word2vec model loss.
Parses command-line arguments, loads data, trains the model, and plots the training loss.
"""

import argparse
import matplotlib.pyplot as plt
from annotix_ml.word2vec import process_text, train

parser = argparse.ArgumentParser(description="Train a word2vec model.")
parser.add_argument("--context_size", type=int, default=2, help="Size of context window")
parser.add_argument("--embedding_dim", type=int, default=10, help="Dimension of word embeddings")
parser.add_argument("--algo", type=str, choices=["ngram", "cbow"], default="ngram", help="Algorithm: 'ngram' or 'cbow'")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
parser.add_argument("--device", type=str, default="mps", help="GPU device to use (e.g., 'cpu', 'cuda', 'mps')")
args = parser.parse_args()

CONTEXT_SIZE = args.context_size
EMBEDDING_DIM = args.embedding_dim
ALGO = args.algo
BATCH_SIZE = args.batch_size
DEVICE = args.device

# Load example text
with open("main/test_data/shakespeare.txt", "r") as f:
    text = f.read().split()

ngrams, vocab, word_to_ix = process_text(text, CONTEXT_SIZE, ALGO)

model, losses = train(ngrams=ngrams, vocab=vocab, word_to_ix=word_to_ix, embedding_dim=EMBEDDING_DIM, 
                        context_size=CONTEXT_SIZE, epochs=100, device=DEVICE, batch_size=BATCH_SIZE, algo=ALGO)

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.show()
