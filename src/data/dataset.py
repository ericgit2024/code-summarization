from datasets import load_dataset
import ast
import logging

logger = logging.getLogger(__name__)

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
    Loads the custom dataset and filters it.
    
    Args:
        split (str): Dataset split to load ('train', 'validation', 'test')
    
    Returns:
        Dataset: Filtered dataset
    
    If split is "validation" or "test" and not found, it splits the "train" set.
    """
    logger.info(f"Loading custom dataset, split: {split}")
    
    # Load custom dataset
    try:
        dataset = load_dataset("json", data_files="code_summary_dataset.jsonl", split=split)
    except ValueError:
        # Fallback: if validation/test split not found, split the train set
        logger.info(f"Split '{split}' not found. Splitting 'train' dataset to create it.")
        full_dataset = load_dataset("json", data_files="code_summary_dataset.jsonl", split="train")
        # Deterministic split for consistency
        dataset_split = full_dataset.train_test_split(test_size=0.1, seed=42)

        if split == "validation" or split == "test":
            dataset = dataset_split["test"]
        else:
            dataset = dataset_split["train"]

    dataset = dataset.filter(is_valid_example)
    
    # Log dataset statistics
    logger.info(f"Dataset statistics after filtering:")
    logger.info(f"  Total examples: {len(dataset)}")
    if len(dataset) > 0:
        avg_code_len = sum(len(ex['code']) for ex in dataset) / len(dataset)
        avg_summary_len = sum(len(ex['summary']) for ex in dataset) / len(dataset)
        logger.info(f"  Average code length: {avg_code_len:.1f} chars")
        logger.info(f"  Average summary length: {avg_summary_len:.1f} chars")
    
    return dataset
