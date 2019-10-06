import numpy as np
import csv
import os
import pickle
from fiction_segmentation import const


def clean_encode_indices(text, w2v_data_path=const.W2V_DATA_PATH):
    _, word2idx = load_vocab(w2v_data_path)

    encoded_text = []
    # For each word in sentence, replace by number
    for sent in text:
        encoded_text.append([word2idx[word] for word in sent if word in word2idx.keys()])

    return encoded_text


def sent_embed_encode(sent, w2v_data_path=const.W2V_DATA_PATH):
    wv = load_embeddings(w2v_data_path)

    # For each word in sentence, return embeddings
    encoded_list = []
    for word in sent:
        encoded_list.append(wv[word])

    return np.array(encoded_list)


def save_vocab(vocab, export_path):
    """Save vocabulary as a dictionary and a list"""
    # Generate objects
    idx2word = vocab.keys()
    word2idx = {word: idx for idx, word in enumerate(idx2word)}

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
