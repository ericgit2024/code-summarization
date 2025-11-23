from src.data.dataset import load_and_process_dataset
from src.retrieval.rag import RAGSystem
import os
import pickle

def index_dataset(index_path="rag_index.pkl"):
    print("Loading dataset...")
    dataset = load_and_process_dataset(split="train")

    # For demonstration, use a smaller subset to speed up indexing
    # In production, remove the slicing
    subset_size = 1000
    print(f"Indexing first {subset_size} examples...")
    codes = dataset['code'][:subset_size]
    docstrings = dataset['docstring'][:subset_size]

    print("Initializing RAG system...")
    rag_system = RAGSystem()

    print("Building index...")
    rag_system.build_index(codes, docstrings)

    print(f"Saving index to {index_path}...")
    with open(index_path, "wb") as f:
        pickle.dump(rag_system, f)

    print("Indexing complete.")

if __name__ == "__main__":
    index_dataset()
