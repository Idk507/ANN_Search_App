import faiss
import numpy as np

class ANNIndexer:
    def __init__(self, vectors, method="flat"):
        self.vectors = vectors.astype('float32')
        self.dim = self.vectors.shape[1]
        self.method = method.lower()
        self.index = self.build_index()

    def build_index(self):
        if self.method == "flat":
            index = faiss.IndexFlatL2(self.dim)
        elif self.method == "hnsw":
            index = faiss.IndexHNSWFlat(self.dim, 32)
            index.hnsw.efConstruction = 40
            index.hnsw.efSearch = 16
        elif self.method == "ivfpq":
            quantizer = faiss.IndexFlatL2(self.dim)
            index = faiss.IndexIVFPQ(quantizer, self.dim, 100, 8, 8)
            index.train(self.vectors)
            index.nprobe = 10
        else:
            raise ValueError("Unsupported index method")
        
        index.add(self.vectors)
        return index

    def search(self, query_vector, k=5):
        D, I = self.index.search(np.array([query_vector], dtype='float32'), k)
        return D[0], I[0]
