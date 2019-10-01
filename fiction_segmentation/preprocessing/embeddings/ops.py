import numpy as np
import csv
import os
import pickle

def clean_encode_indices():
    return


def encode_embdeddings():
    return


def save_vocab(vocab, export_path):
    """Save vocabulary as a dictionary and a list"""
    # Generate objects
    idx2word = vocab.keys()
    word2idx = {word:idx for idx, word in enumerate(idx2word)}

    # Save list as CSV
    with open(os.path.join(export_path, 'idx2word.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        for word in idx2word:
            writer.writerow([word])

    # Save dictionary using pickle
    with open(os.path.join(export_path, 'word2idx.pickle'), mode='wb') as file:
        pickle.dump(word2idx, file)


def load_vocab(import_path):
    # Load idx2word
    with open(os.path.join(import_path, 'idx2word.csv'), 'r') as file:
        idx2word = [word[0] for word in list(csv.reader(file))]

    # Load word2idx
    with open(os.path.join(import_path, 'word2idx.pickle'), 'rb') as file:
        word2idx = pickle.load(file)

    return idx2word, word2idx


def save_embeddings(wv, export_path):
    # Generate dictionary from word vectors
    embeddings = np.array([wv[word] for word in wv.vocab.keys()])

    # Save embeddings
    np.save(os.path.join(export_path, 'embeddings.npy'), embeddings)


def load_embeddings(import_path):
    # Read embeddings
    return np.load(os.path.join(import_path, 'embeddings.npy'))
