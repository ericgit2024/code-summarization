import sys
import os
import pickle

# Add the project root to sys.path to resolve 'src' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.retrieval.rag import RAGSystem
from src.data.dataset import load_and_process_dataset

def build_index(
    dataset_split="train",
    output_path="rag_index.pkl"
):
    """
    Builds a RAG index from the specified dataset split and saves it to a file.
    """
    print("Loading and preparing dataset for RAG index...")
    dataset = load_and_process_dataset(split=dataset_split)

    print("Building RAG index...")
    rag_system = RAGSystem()

    # Extract codes and summaries from the dataset
    codes = [example['code'] for example in dataset]
    summaries = [example['summary'] for example in dataset]

    rag_system.build_index(codes, summaries)

    print(f"Saving RAG index to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(rag_system, f)

    print("RAG index build complete.")

if __name__ == "__main__":
    build_index()
