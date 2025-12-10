from datasets import load_dataset
import ast
import logging
import re

logger = logging.getLogger(__name__)

def clean_text(text):
    """
    Cleans text by removing excessive whitespace and newlines.
    """
    if not text:
        return ""
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n+', '\n', text)
    # Replace multiple spaces with a single space
    text = re.sub(r' +', ' ', text)
    return text.strip()

def is_valid_example(example):
    """
    Checks if the example has valid python code and a non-empty summary.
    Also filters by length.
    """
    code = example.get('code', '')
    summary = example.get('summary', '')

    if not code or not summary:
        return False

    # Check lengths
    # Simple token estimation by splitting by whitespace
    code_len = len(code.split())
    summary_len = len(summary.split())

    if code_len < 10 or code_len > 500:
        return False

    if summary_len < 3 or summary_len > 512:
        return False

    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

def load_and_process_dataset(split="train"):
    """
    Loads the CodeSearchNet (code_x_glue_ct_code_to_text) dataset and filters it.
    
    Args:
        split (str): Dataset split to load. Currently only uses 'train' logic
                     as we split it manually later, but supports loading
                     a specific split if needed for other purposes.
    
    Returns:
        Dataset: Filtered and mapped dataset
    """
    logger.info(f"Loading CodeSearchNet (code_x_glue_ct_code_to_text) dataset, split: {split}")
    
    # We load the dataset
    # Loading ~25k examples from train to ensure we have enough after filtering
    # to meet the requirement of 10k-15k train + 1.5k-2k val (total ~17k)
    try:
        # Load Python subset
        # We assume usage of 'train' mostly.
        if split == "train":
            # Load enough data to filter down to ~18k. Loading full train set (250k) is fast enough.
            dataset = load_dataset("code_x_glue_ct_code_to_text", "python", split="train", trust_remote_code=True)
        else:
            # For validation/test, we might still want to use the 'train' split if we are doing
            # the custom split in trainer.py. But if the user asks for 'validation' directly
            # we can try to load it.
            # However, the requirement says "Replace with new split on CodeSearchNet"
            # and "Split ratio: 85% train / 15% validation".
            # So we stick to loading from train and let the trainer split it.
            # But just in case, we load 'train' as default source.
            dataset = load_dataset("code_x_glue_ct_code_to_text", "python", split="train", trust_remote_code=True)

    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        # Fallback to custom dataset if internet fails (though instructions say REPLACE)
        raise e

    # Map fields
    # CodeSearchNet fields: code, docstring
    # Our fields: code, summary
    def map_fields(example):
        return {
            "code": clean_text(example.get("code", "")),
            "summary": clean_text(example.get("docstring", ""))
        }

    dataset = dataset.map(map_fields)

    # Filter
    dataset = dataset.filter(is_valid_example)
    
    # Limit to ~18000 to satisfy the size constraint (approx 15k train + 2.7k val)
    if len(dataset) > 18000:
        dataset = dataset.select(range(18000))

    # Log dataset statistics
    logger.info(f"Dataset statistics after filtering:")
    logger.info(f"  Total examples: {len(dataset)}")
    if len(dataset) > 0:
        avg_code_len = sum(len(ex['code']) for ex in dataset) / len(dataset)
        avg_summary_len = sum(len(ex['summary']) for ex in dataset) / len(dataset)
        logger.info(f"  Average code length: {avg_code_len:.1f} chars")
        logger.info(f"  Average summary length: {avg_summary_len:.1f} chars")
    
    return dataset
