from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class RAGSystem:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.encoder_model = SentenceTransformer(model_name)
        self.index = None
        self.stored_codes = []
        self.stored_docstrings = []

    def encode_codes(self, code_list):
        return self.encoder_model.encode(code_list, convert_to_numpy=True)

    def build_index(self, codes, docstrings):
        """
        Builds the FAISS index with the provided codes and docstrings.
        """
        embeddings = self.encode_codes(codes)
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(embeddings)
        self.stored_codes = codes
        self.stored_docstrings = docstrings

    def retrieve(self, query_code, k=5):
        """
        Retrieves the top-k similar codes and docstrings for a given query code.
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")

        query_embedding = self.encode_codes([query_code])
        distances, indices = self.index.search(query_embedding, k)
        retrieved_indices = indices[0].tolist()

        retrieved_codes = [self.stored_codes[i] for i in retrieved_indices]
        retrieved_docstrings = [self.stored_docstrings[i] for i in retrieved_indices]

        return retrieved_codes, retrieved_docstrings, distances[0]
