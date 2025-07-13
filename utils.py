from sentence_transformers import SentenceTransformer
import numpy as np

def load_vectors():
    # Simulate 5000 random 128D vectors
    return np.random.random((5000, 128)).astype('float32')

def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

def encode_text(text, model):
    return model.encode([text])[0]
