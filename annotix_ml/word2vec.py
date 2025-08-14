"""
Word2Vec Training Module

This module provides implementations for n-gram and continuous bag-of-words (CBOW) word2vec models using PyTorch.
It includes utilities for processing text, batching data, and training the models.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

#TODO add word_to_idx in the objects
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

    def __init__(self, vocab_size, embedding_dim, context_size, word_to_ix):
        super(NGramLanguageModeler, self).__init__()
        
        # Attributes
        self.vocab_size = vocab_size
        self.context_size = context_size  # Number of context words
        self.embedding_dim = embedding_dim  # Dimension of word embeddings
        self.word_to_ix = word_to_ix  # Store the mapping for later use

        # Layers
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

    def __init__(self, vocab_size, embedding_dim, word_to_ix, context_size=None):
        super(CBOW, self).__init__()

        # Attributes
        self.vocab_size = vocab_size
        self.context_size = context_size  # Number of context words
        self.embedding_dim = embedding_dim  # Dimension of word embeddings
        self.word_to_ix = word_to_ix  # Store the mapping for later use

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)
        self.word_to_ix = word_to_ix

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

def cosine_similarity(vec1, vec2):
    """
    Computes cosine similarity between two vectors.

    Args:
        vec1 (Tensor): First vector.
        vec2 (Tensor): Second vector.

    Returns:
        float: Cosine similarity value.
    """
    if not isinstance(vec1, torch.Tensor):
        vec1 = torch.tensor(vec1, dtype=torch.float32)
    if not isinstance(vec2, torch.Tensor):
        vec2 = torch.tensor(vec2, dtype=torch.float32)
    cos = nn.CosineSimilarity(dim=0)
    return cos(vec1, vec2).item()

def word_similarity(model, word1, word2):
    """
    Computes similarity between two words using the trained model.

    Args:
        model (nn.Module): Trained word2vec model.
        word1 (str): First word.
        word2 (str): Second word.
        word_to_ix (dict): Mapping from word to index.

    Returns:
        float: Similarity score between the two words.
    """
    if hasattr(model, "wv"):
        vec1 = model.wv[word1]
        vec2 = model.wv[word2]
    else:
        idx1 = model.word_to_ix.get(word1)
        idx2 = model.word_to_ix.get(word2)

        if not idx1 or not idx2:
            raise ValueError(f"Words '{word1}' or '{word2}' not found in vocabulary.")

        vec1 = model.embeddings.weight[idx1].cpu()
        vec2 = model.embeddings.weight[idx2].cpu()

    return cosine_similarity(torch.tensor(vec1), torch.tensor(vec2))

def embedding(model, word):
    """
    Retrieves the embedding vector for a given word from the specified model.

    Args:
        model: An object containing word embeddings and a mapping from words to indices.
        word (str): The word for which to obtain the embedding vector.

    Returns:
        numpy.ndarray: The embedding vector corresponding to the given word.

    Raises:
        KeyError: If the word is not present in the model's vocabulary.
    """
    indexes = model.word_to_ix[word]
    return model.embeddings.weight[indexes].detach().numpy()

def spectrum_simularity(model, spec_doc_1, spec_doc_2):
    """
    Computes the cosine similarity between the average word embeddings of two spectrum documents.

    Args:
        model: The word embedding model used to generate embeddings for words.
        spec_doc_1: An object representing the first spectrum document, expected to have a 'words' attribute.
        spec_doc_2: An object representing the second spectrum document, expected to have a 'words' attribute.

    Returns:
        float: The cosine similarity between the averaged embeddings of the two spectrum documents.
    """
    vec_spec1 = np.mean([embedding(model, word) for word in spec_doc_1.words], axis=0)
    vec_spec2 = np.mean([embedding(model, word) for word in spec_doc_2.words], axis=0)
    return cosine_similarity(vec_spec1, vec_spec2)

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
        model = CBOW(vocab_size=len(vocab), embedding_dim=embedding_dim, word_to_ix=word_to_ix)
    else:
        model = NGramLanguageModeler(vocab_size=len(vocab), embedding_dim=embedding_dim, word_to_ix=word_to_ix, context_size=context_size)

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
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    return model, losses
