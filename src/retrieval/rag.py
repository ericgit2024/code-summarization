from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import logging

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, model_name='microsoft/codebert-base'):
        """
        Initializes RAG system with a code-specific embedding model.
        Falls back to all-MiniLM if codebert is not available/fails (though sentence-transformers handles download).
        """
        try:
            self.encoder_model = SentenceTransformer(model_name)
            logger.info(f"Loaded RAG model: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load {model_name}, falling back to all-MiniLM-L6-v2. Error: {e}")
            self.encoder_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        self.index = None
        self.stored_codes = []
        self.stored_metadata = [] # Renamed from docstrings to metadata to store more info

    def encode_codes(self, code_list, metadata_list=None):
        """
        Encodes code snippets.
        Optionally augments code with metadata (e.g. function name, docstring) before encoding.
        """
        texts_to_encode = []
        for i, code in enumerate(code_list):
            text = code
            if metadata_list and i < len(metadata_list):
                meta = metadata_list[i]
                # Augment with function name and docstring for better semantic matching
                func_name = meta.get("name", "")
                doc = meta.get("docstring", "")
                if func_name or doc:
                    text = f"{func_name}\n{doc}\n{code}"
            texts_to_encode.append(text)

        return self.encoder_model.encode(texts_to_encode, convert_to_numpy=True)

    def build_index(self, codes, metadata_list):
        """
        Builds the FAISS index.
        metadata_list: list of dicts containing 'docstring', 'name', 'complexity', etc.
        """
        logger.info(f"Building RAG index with {len(codes)} items...")
        embeddings = self.encode_codes(codes, metadata_list)
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(embeddings)
        self.stored_codes = codes
        self.stored_metadata = metadata_list

    def retrieve(self, query_code, k=5, diversity_penalty=0.5):
        """
        Retrieves top-k similar codes.
        Applies Maximal Marginal Relevance (MMR) like logic or simple reranking for diversity.
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")

        # Retrieve more candidates first to allow for filtering/reranking
        fetch_k = k * 3
        query_embedding = self.encode_codes([query_code])[0]
        distances, indices = self.index.search(np.array([query_embedding]), fetch_k)

        candidates_indices = indices[0].tolist()
        candidates_distances = distances[0].tolist()

        # Simple diversity reranking: avoid very similar results
        # We can check similarity between candidates.
        # But for simplicity and speed, let's filter by uniqueness of function names or file paths if available
        # OR use a simple distance threshold check if we had embeddings of candidates available easily.
        # Since FAISS gives us distances to query, we rely on that for relevance.
        # Diversity: ensure we don't return 3 variations of the same function name from different files?

        selected_indices = []
        seen_names = set()

        for idx, dist in zip(candidates_indices, candidates_distances):
            if idx == -1: continue

            meta = self.stored_metadata[idx]
            name = meta.get("name", "unknown")

            # Diversity check: don't select same function name multiple times
            if name in seen_names:
                continue

            seen_names.add(name)
            selected_indices.append(idx)

            if len(selected_indices) >= k:
                break

        retrieved_codes = [self.stored_codes[i] for i in selected_indices]
        retrieved_metadata = [self.stored_metadata[i] for i in selected_indices]
        # Distances for selected
        # We need to map back to distances... let's just return a placeholder or recalculate if needed.
        # For now, just return selected items.

        return retrieved_codes, retrieved_metadata, [] # Distances ignored for now
