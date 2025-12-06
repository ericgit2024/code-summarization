"""
Create Train/Validation/Test Splits for CodeXGlue Dataset

Creates stratified splits based on code complexity to ensure balanced distribution.

Usage:
    python -m src.scripts.create_dataset_splits --input codexglue_processed.jsonl
"""

import argparse
import json
import logging
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_dataset(input_file):
    """Load JSONL dataset into memory."""
    logger.info(f"Loading dataset from: {input_file}")
    
    examples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line))
    
    logger.info(f"Loaded {len(examples)} examples")
    return examples


def get_complexity_bin(complexity):
    """
    Bin complexity values for stratification.
    
    Args:
        complexity (int): Cyclomatic complexity value
    
    Returns:
        str: Complexity bin label
    """
    if complexity <= 3:
        return 'low'
    elif complexity <= 10:
        return 'medium'
    else:
        return 'high'


def create_splits(examples, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Create stratified train/validation/test splits.
    
    Args:
        examples (list): List of example dictionaries
        train_ratio (float): Proportion for training set
        val_ratio (float): Proportion for validation set
        test_ratio (float): Proportion for test set
        seed (int): Random seed for reproducibility
    
    Returns:
        tuple: (train_examples, val_examples, test_examples)
    """
    logger.info(f"Creating splits: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "Ratios must sum to 1.0"
    
    # Extract complexity bins for stratification
    complexity_bins = [get_complexity_bin(ex.get('complexity', 1)) for ex in examples]
    
    # Log complexity distribution
    bin_counts = Counter(complexity_bins)
    logger.info(f"Complexity distribution:")
    for bin_name, count in sorted(bin_counts.items()):
        logger.info(f"  {bin_name}: {count} ({count/len(examples)*100:.1f}%)")
    
    # First split: separate test set
    train_val_examples, test_examples, train_val_bins, test_bins = train_test_split(
        examples,
        complexity_bins,
        test_size=test_ratio,
        random_state=seed,
        stratify=complexity_bins
    )
    
    # Second split: separate validation from training
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    train_examples, val_examples, train_bins, val_bins = train_test_split(
        train_val_examples,
        train_val_bins,
        test_size=val_size_adjusted,
        random_state=seed,
        stratify=train_val_bins
    )
    
    logger.info(f"\nSplit sizes:")
    logger.info(f"  Train: {len(train_examples)} examples")
    logger.info(f"  Validation: {len(val_examples)} examples")
    logger.info(f"  Test: {len(test_examples)} examples")
    
    # Verify stratification
    logger.info(f"\nComplexity distribution per split:")
    for split_name, split_examples in [('Train', train_examples), ('Val', val_examples), ('Test', test_examples)]:
        split_bins = [get_complexity_bin(ex.get('complexity', 1)) for ex in split_examples]
        split_counts = Counter(split_bins)
        logger.info(f"  {split_name}:")
        for bin_name in ['low', 'medium', 'high']:
            count = split_counts.get(bin_name, 0)
            pct = count / len(split_examples) * 100 if split_examples else 0
            logger.info(f"    {bin_name}: {count} ({pct:.1f}%)")
    
    return train_examples, val_examples, test_examples


def check_data_leakage(train_examples, val_examples, test_examples):
    """
    Check for data leakage between splits.
    
    Args:
        train_examples (list): Training examples
        val_examples (list): Validation examples
        test_examples (list): Test examples
    
    Returns:
        bool: True if no leakage detected
    """
    logger.info("Checking for data leakage...")
    
    # Create sets of code hashes for comparison
    train_codes = set(hash(ex['code']) for ex in train_examples)
    val_codes = set(hash(ex['code']) for ex in val_examples)
    test_codes = set(hash(ex['code']) for ex in test_examples)
    
    # Check for overlaps
    train_val_overlap = train_codes & val_codes
    train_test_overlap = train_codes & test_codes
    val_test_overlap = val_codes & test_codes
    
    if train_val_overlap:
        logger.warning(f"⚠ Found {len(train_val_overlap)} overlapping examples between train and validation")
    if train_test_overlap:
        logger.warning(f"⚠ Found {len(train_test_overlap)} overlapping examples between train and test")
    if val_test_overlap:
        logger.warning(f"⚠ Found {len(val_test_overlap)} overlapping examples between validation and test")
    
    if not (train_val_overlap or train_test_overlap or val_test_overlap):
        logger.info("✓ No data leakage detected")
        return True
    else:
        logger.error("✗ Data leakage detected!")
        return False


def save_split(examples, output_file):
    """Save a split to JSONL file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    logger.info(f"Saved {len(examples)} examples to: {output_path.absolute()}")


def main():
    parser = argparse.ArgumentParser(
        description="Create train/validation/test splits for CodeXGlue dataset"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input JSONL file (preprocessed data)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='Output directory for split files (default: current directory)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Training set ratio (default: 0.8)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Validation set ratio (default: 0.1)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='Test set ratio (default: 0.1)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    try:
        # Load dataset
        examples = load_dataset(args.input)
        
        # Create splits
        train_examples, val_examples, test_examples = create_splits(
            examples,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        )
        
        # Check for data leakage
        no_leakage = check_data_leakage(train_examples, val_examples, test_examples)
        
        if not no_leakage:
            logger.error("Aborting due to data leakage")
            return 1
        
        # Save splits
        output_dir = Path(args.output_dir)
        save_split(train_examples, output_dir / 'codexglue_train.jsonl')
        save_split(val_examples, output_dir / 'codexglue_validation.jsonl')
        save_split(test_examples, output_dir / 'codexglue_test.jsonl')
        
        logger.info("\n✓ Splits created successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"\n✗ Split creation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
