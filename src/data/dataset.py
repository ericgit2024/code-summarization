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
    """
    dataset = load_dataset("json", data_files="code_summary_dataset.jsonl", split=split)
    dataset = dataset.filter(is_valid_example)
    return dataset
