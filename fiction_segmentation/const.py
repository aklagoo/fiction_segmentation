# Constants for preprocessing.embeddings.Word2Vec
W2V_EMBEDDING_DIM = 300
W2V_MIN_WORD_COUNT = 3
W2V_CONTEXT_SIZE = 7
W2V_DOWNSAMPLING = 1e-3
W2V_SEED = 1
W2V_DATA_PATH = 'data/embeddings'
W2V_EPOCHS = 10

# Global constants
SAMPLE_SIZE = 50
NUM_CLASSES = 13
CLASS_IDX = {
    'tranquil': 0,
    'joy': 1,
    'sad': 2,
    'angry': 3,
    'tense': 4,
    'funny': 5,
    'shock': 6,
    'romantic': 7,
    'serious': 8,
    'peaceful': 9,
    'fear': 10,
    'pain': 11,
    'happy': 12
}
