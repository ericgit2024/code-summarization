from datasets import load_dataset
import ast

def is_valid_example(example):
    """
    Checks if the example has valid python code and a non-empty summary.
    """
    try:
        ast.parse(example['code'])
        return len(example['summary'].strip()) > 0
    except SyntaxError:
        return False

def load_and_process_dataset(split="train"):
    """
    Loads the custom code_summary_dataset.jsonl dataset and filters it.

    If split is "validation" or "test" and not found, it splits the "train" set.
    """
    try:
        dataset = load_dataset("json", data_files="code_summary_dataset.jsonl", split=split)
    except ValueError:
        # Fallback: if validation/test split not found, split the train set
        print(f"Split '{split}' not found. Splitting 'train' dataset to create it.")
        full_dataset = load_dataset("json", data_files="code_summary_dataset.jsonl", split="train")
        # Deterministic split for consistency
        dataset_split = full_dataset.train_test_split(test_size=0.1, seed=42)

        if split == "validation" or split == "test":
            dataset = dataset_split["test"]
        else:
            dataset = dataset_split["train"]

    dataset = dataset.filter(is_valid_example)
    return dataset
