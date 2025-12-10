from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import logging

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initializes RAG system with a simple, reliable embedding model.
        Using all-MiniLM-L6-v2 for stability (UniXCoder had CUDA issues).
        """
        try:
            self.encoder_model = SentenceTransformer(model_name)
            logger.info(f"Loaded RAG model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load {model_name}. Error: {e}")
            raise

        self.index = None
        self.stored_codes = []
        self.stored_metadata = [] # Renamed from docstrings to metadata to store more info

    def __setstate__(self, state):
        """
        Handle legacy pickle files that might not have stored_metadata or have old attribute names.
        """
        self.__dict__.update(state)
        # Check if stored_metadata is missing
        if not hasattr(self, 'stored_metadata'):
            # Try to restore from old attributes if they exist
            if hasattr(self, 'docstrings'):
                logger.info("Migrating legacy 'docstrings' attribute to 'stored_metadata'")
                # Convert list of strings to list of dicts
                self.stored_metadata = [{'docstring': d} for d in self.docstrings]
                # Optional: delete old attribute if you want to clean up
                # del self.docstrings
            elif hasattr(self, 'stored_docstrings'):
                logger.info("Migrating legacy 'stored_docstrings' attribute to 'stored_metadata'")
                self.stored_metadata = [{'docstring': d} for d in self.stored_docstrings]
            else:
                logger.warning("No metadata found in legacy pickle. Initializing empty metadata.")
                self.stored_metadata = [{} for _ in self.stored_codes] if hasattr(self, 'stored_codes') else []

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

        # Check if indices is a numpy array or list (depends on FAISS version and mocking)
        if isinstance(indices, np.ndarray):
            candidates_indices = indices[0].tolist()
        else:
            candidates_indices = indices[0]

        if isinstance(distances, np.ndarray):
            candidates_distances = distances[0].tolist()
        else:
            candidates_distances = distances[0]

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

            # Handle case where index might be out of bounds (should not happen if consistent)
            if idx >= len(self.stored_metadata):
                logger.warning(f"Retrieved index {idx} is out of bounds for stored_metadata (len={len(self.stored_metadata)}). Skipping.")
                continue

            meta = self.stored_metadata[idx]
            # Ensure meta is a dict (handle legacy string data if migration failed or mixed)
            if isinstance(meta, str):
                 meta = {'docstring': meta}

            name = meta.get("name", "unknown")

            # Diversity check: don't select same function name multiple times
            if name in seen_names:
                continue

            seen_names.add(name)
            selected_indices.append(idx)

            if len(selected_indices) >= k:
                break

        retrieved_codes = [self.stored_codes[i] for i in selected_indices]
        # Normalize metadata to dicts in output as well
        retrieved_metadata = []
        for i in selected_indices:
             m = self.stored_metadata[i]
             if isinstance(m, str):
                 m = {'docstring': m}
             retrieved_metadata.append(m)

        # Distances for selected
        # We need to map back to distances... let's just return a placeholder or recalculate if needed.
        # For now, just return selected items.

        return retrieved_codes, retrieved_metadata, [] # Distances ignored for now
