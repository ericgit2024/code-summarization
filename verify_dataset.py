from src.data.dataset import load_and_process_dataset
import os

# Create a dummy dataset file if it doesn't exist for testing purposes
if not os.path.exists("code_summary_dataset.jsonl"):
    with open("code_summary_dataset.jsonl", "w") as f:
        f.write('{"code": "def foo(): pass", "docstring": "A function"}\n')

try:
    ds = load_and_process_dataset()
    print("Dataset loaded successfully!")
    print(ds)
except Exception as e:
    print(f"Failed to load dataset: {e}")
