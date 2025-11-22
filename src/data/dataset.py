from datasets import load_dataset
import ast

def is_valid_example(example):
    """
    Checks if the example has valid python code and a non-empty docstring.
    """
    try:
        ast.parse(example['code'])
        return len(example['docstring'].strip()) > 0
    except SyntaxError:
        return False

def load_and_process_dataset(split="train"):
    """
    Loads the code_x_glue_ct_code_to_text dataset for python and filters it.
    """
    dataset = load_dataset("code_x_glue_ct_code_to_text", "python", split=split)
    dataset = dataset.filter(is_valid_example)
    return dataset
