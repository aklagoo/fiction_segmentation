import gensim.models.word2vec as word2vec
from fiction_segmentation import const
from fiction_segmentation import preprocessing
import multiprocessing
import os
import pickle


class Word2Vec:
    """word2vec trainable model from 'gensim'
    """

    def __init__(self, sg=1, seed=const.W2V_SEED, workers=multiprocessing.cpu_count(),
                 size=const.W2V_EMBEDDING_DIM, min_count=const.W2V_MIN_WORD_COUNT,
                 window=const.W2V_CONTEXT_SIZE, sample=const.W2V_DOWNSAMPLING):
        # Generate model
        self.model = word2vec.Word2Vec(sg=sg, seed=seed, workers=workers, size=size, min_count=min_count,
                                       window=window, sample=sample)

    def generate_embeddings(self, text, epochs=const.W2V_EPOCHS, export_path=const.W2V_DATA_PATH):
        sentences = preprocessing.stem(preprocessing.tokenize(text))
        self.model.build_vocab(sentences)
        self.model.train(sentences, total_examples=self.model.corpus_count, epochs=epochs)

        # Save data to file
        preprocessing.embeddings.save_vocab(self.model.wv.vocab, export_path)
        preprocessing.embeddings.save_embeddings(self.model.wv, export_path)
