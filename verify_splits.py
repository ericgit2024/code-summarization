"""
Validation script for dataset splits.

Usage:
    python verify_splits.py train.jsonl validation.jsonl test.jsonl
"""

import json
import sys
from collections import Counter

def load_dataset(file_path):
    """Load JSONL dataset."""
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line))
    return examples

def verify_splits(train_file, val_file, test_file):
    """Verify dataset splits for quality and no data leakage."""
    
    print("Verifying dataset splits...")
    print("="*60)
    
    # Load datasets
    print("Loading datasets...")
    train_data = load_dataset(train_file)
    val_data = load_dataset(val_file)
    test_data = load_dataset(test_file)
    
    print(f"Train: {len(train_data)} examples")
    print(f"Validation: {len(val_data)} examples")
    print(f"Test: {len(test_data)} examples")
    print(f"Total: {len(train_data) + len(val_data) + len(test_data)} examples")
    
    # Check for data leakage
    print("\nChecking for data leakage...")
    train_codes = set(hash(ex['code']) for ex in train_data)
    val_codes = set(hash(ex['code']) for ex in val_data)
    test_codes = set(hash(ex['code']) for ex in test_data)
    
    train_val_overlap = train_codes & val_codes
    train_test_overlap = train_codes & test_codes
    val_test_overlap = val_codes & test_codes
    
    leakage_found = False
    if train_val_overlap:
        print(f"❌ Found {len(train_val_overlap)} overlapping examples between train and validation")
        leakage_found = True
    if train_test_overlap:
        print(f"❌ Found {len(train_test_overlap)} overlapping examples between train and test")
        leakage_found = True
    if val_test_overlap:
        print(f"❌ Found {len(val_test_overlap)} overlapping examples between validation and test")
        leakage_found = True
    
    if not leakage_found:
        print("✅ No data leakage detected")
    
    # Check complexity distribution
    print("\nComplexity distribution:")
    
    def get_complexity_bin(complexity):
        if complexity <= 3:
            return 'low'
        elif complexity <= 10:
            return 'medium'
        else:
            return 'high'
    
    for split_name, split_data in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
        bins = [get_complexity_bin(ex.get('complexity', 1)) for ex in split_data]
        bin_counts = Counter(bins)
        print(f"  {split_name}:")
        for bin_name in ['low', 'medium', 'high']:
            count = bin_counts.get(bin_name, 0)
            pct = count / len(split_data) * 100 if split_data else 0
            print(f"    {bin_name}: {count} ({pct:.1f}%)")
    
    # Check summary length distribution
    print("\nAverage summary length:")
    for split_name, split_data in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
        avg_len = sum(len(ex['summary']) for ex in split_data) / len(split_data) if split_data else 0
        print(f"  {split_name}: {avg_len:.1f} chars")
    
    print("="*60)
    
    # Final verdict
    if not leakage_found:
        print("✅ PASS: All validation checks passed")
        return True
    else:
        print("❌ FAIL: Data leakage detected")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python verify_splits.py <train.jsonl> <val.jsonl> <test.jsonl>")
        sys.exit(1)
    
    train_file, val_file, test_file = sys.argv[1], sys.argv[2], sys.argv[3]
    passed = verify_splits(train_file, val_file, test_file)
    sys.exit(0 if passed else 1)
