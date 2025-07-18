"""
Word2Vec Training Module

This module provides implementations for n-gram and continuous bag-of-words (CBOW) word2vec models using PyTorch.
It includes utilities for processing text, batching data, and training the models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm


class NGramLanguageModeler(nn.Module):
    """
    Neural network for n-gram language modeling.

    Args:
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimension of word embeddings.
        context_size (int): Number of context words.

    Methods:
        forward(inputs): Forward pass for batch of context word indices.
    """

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        """
        Forward pass of the n-gram model.

        Args:
            inputs (Tensor): Batch of context word indices of shape (batch_size, context_size).

        Returns:
            Tensor: Log probabilities for each word in the vocabulary.
        """
        embeds = self.embeddings(inputs).view((inputs.size(0), -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

class CBOW(nn.Module):
    """
    Continuous Bag-of-Words (CBOW) model.

    Args:
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimension of word embeddings.
        context_size (int): Number of context words.

    Methods:
        forward(inputs): Forward pass for batch of context word indices.
    """

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        """
        Forward pass of the CBOW model.

        Args:
            inputs (Tensor): Batch of context word indices of shape (batch_size, context_size).

        Returns:
            Tensor: Log probabilities for each word in the vocabulary.
        """
        embeds = self.embeddings(inputs)  # (batch_size, context_size, embedding_dim)
        avg_embeds = embeds.mean(dim=1)  # (batch_size, embedding_dim)
        out = F.relu(self.linear1(avg_embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

def process_text(text, context_size, type="ngram"):
    """
    Processes raw text into n-grams or CBOW data and builds vocabulary.

    Args:
        text (list of str): Tokenized text.
        context_size (int): Number of context words.
        type (str): 'ngram' or 'cbow' to select algorithm.

    Returns:
        tuple: (ngrams/data, vocab, word_to_ix)
            ngrams/data (list): List of (context, target) pairs.
            vocab (set): Set of unique words.
            word_to_ix (dict): Mapping from word to index.
    """
    if type == "ngram":
        return process_text_ngram(text, context_size)
    elif type == "cbow":
        return process_text_cbow(text, context_size)
    else:
        raise ValueError("Unknown type. Use 'ngram' or 'cbow'.")

def process_text_ngram(text, context_size):
    """
    Generates n-gram (context, target) pairs from text.

    Args:
        text (list of str): Tokenized text.
        context_size (int): Number of context words.

    Returns:
        tuple: (ngrams, vocab, word_to_ix)
            ngrams (list): List of (context, target) pairs.
            vocab (set): Set of unique words.
            word_to_ix (dict): Mapping from word to index.
    """
    ngrams = [([text[i - j - 1] for j in range(context_size)], text[i]) for i in range(context_size, len(text))]
    # Print the first 3, just so you can see what they look like.
    print(ngrams[:3])

    vocab = set(text)
    word_to_ix = {word: i for i, word in enumerate(vocab)}

    return ngrams, vocab, word_to_ix

def process_text_ngram_2d(sentences, context_size): #TODO to reviews and fix
    """
    Generates n-gram (context, target) pairs from a list of sentences.

    Args:
        sentences (list of list of str): List of tokenized sentences.
        context_size (int): Number of context words.

    Returns:
        tuple: (ngrams, vocab, word_to_ix)
            ngrams (list): List of (context, target) pairs.
            vocab (set): Set of unique words.
            word_to_ix (dict): Mapping from word to index.
    """
    ngrams = []
    vocab = set()
    for text in sentences:
        vocab.update(text)
        for i in range(context_size, len(text)):
            context = [text[i - j - 1] for j in range(context_size)]
            target = text[i]
            ngrams.append((context, target))
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    return ngrams, vocab, word_to_ix

def process_text_cbow(text, context_size):
    """
    Generates CBOW (context, target) pairs from text.

    Args:
        text (list of str): Tokenized text.
        context_size (int): Number of context words on each side.

    Returns:
        tuple: (data, vocab, word_to_ix)
            data (list): List of (context, target) pairs.
            vocab (set): Set of unique words.
            word_to_ix (dict): Mapping from word to index.
    """
    data = []
    for i in range(context_size, len(text) - context_size):
        context = [text[i - j - 1] for j in range(context_size)] + [text[i + j + 1] for j in range(context_size)]
        target = text[i]
        data.append((context, target))

    vocab = set(text)
    word_to_ix = {word: i for i, word in enumerate(vocab)}

    return data, vocab, word_to_ix

def process_text_cbow_2d(sentences, context_size): #TODO to reviews and fix
    """
    Generates CBOW (context, target) pairs from a list of sentences.

    Args:
        sentences (list of list of str): List of tokenized sentences.
        context_size (int): Number of context words.

    Returns:
        tuple: (ngrams, vocab, word_to_ix)
            ngrams (list): List of (context, target) pairs.
            vocab (set): Set of unique words.
            word_to_ix (dict): Mapping from word to index.
    """
    ngrams = []
    vocab = set()
    for text in sentences:
        vocab.update(text)
        for i in range(context_size, len(text)):
            context = [text[i - j - 1] for j in range(context_size)] + [text[i + j + 1] for j in range(context_size)]
            target = text[i]
            ngrams.append((context, target))
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    return ngrams, vocab, word_to_ix

def make_context_vector(context, word_to_ix):
    """
    Converts a context list of words to a tensor of indices.

    Args:
        context (list of str): List of context words.
        word_to_ix (dict): Mapping from word to index.

    Returns:
        Tensor: Tensor of word indices.
    """
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)

def batchify(ngrams, batch_size):
    """
    Splits n-grams or CBOW data into batches.

    Args:
        ngrams (list): List of (context, target) pairs.
        batch_size (int): Size of each batch.

    Yields:
        tuple: (contexts, targets) for each batch.
    """
    for i in range(0, len(ngrams), batch_size):
        batch = ngrams[i:i+batch_size]
        contexts = [context for context, _ in batch]
        targets = [target for _, target in batch]
        yield contexts, targets

def train(ngrams, vocab, word_to_ix, embedding_dim, context_size, epochs=10, device="cpu", batch_size=32, algo="ngram"):
    """
    Trains the n-gram or CBOW language model.

    Args:
        ngrams (list): List of (context, target) pairs.
        vocab (set): Set of unique words.
        word_to_ix (dict): Mapping from word to index.
        embedding_dim (int): Dimension of word embeddings.
        context_size (int): Number of context words.
        epochs (int): Number of training epochs.
        device (str): Device to train on ("cpu", "cuda", "mps").
        batch_size (int): Size of each batch.
        algo (str): 'ngram' or 'cbow' to select model.

    Returns:
        tuple: (model, losses)
            model (nn.Module): Trained model.
            losses (list): Training loss for each epoch.
    """
    losses = []
    loss_function = nn.NLLLoss()
    if algo == "cbow":
        model = CBOW(len(vocab), embedding_dim, context_size)
    else:
        model = NGramLanguageModeler(len(vocab), embedding_dim, context_size)
    model.to(torch.device(device))
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        total_loss = 0
        for contexts, targets in tqdm(batchify(ngrams, batch_size), desc=f"Epoch {epoch + 1}/{epochs}", total=len(ngrams) // batch_size + 1):
            # Prepare batch tensors
            context_idxs = torch.tensor([[word_to_ix[w] for w in context] for context in contexts], dtype=torch.long, device=torch.device(device))
            target_idxs = torch.tensor([word_to_ix[target] for target in targets], dtype=torch.long, device=torch.device(device))

            model.zero_grad()
            log_probs = model(context_idxs)
            loss = loss_function(log_probs, target_idxs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss)
    return model, losses
