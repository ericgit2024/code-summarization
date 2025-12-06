"""
Validation script for preprocessed CodeXGlue dataset.

Usage:
    python verify_preprocessing.py codexglue_processed.jsonl
"""

import json
import sys
import ast
from collections import Counter

def verify_preprocessing(file_path):
    """Verify the preprocessed dataset meets quality standards."""
    
    print(f"Verifying: {file_path}")
    print("="*60)
    
    stats = {
        'total': 0,
        'valid_syntax': 0,
        'has_name': 0,
        'has_complexity': 0,
        'has_summary': 0,
        'empty_summary': 0,
        'short_code': 0,
        'long_code': 0
    }
    
    complexity_dist = Counter()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line)
                stats['total'] += 1
                
                # Check required fields
                code = record.get('code', '')
                summary = record.get('summary', '')
                name = record.get('name', '')
                complexity = record.get('complexity', 0)
                
                # Validate syntax
                try:
                    ast.parse(code)
                    stats['valid_syntax'] += 1
                except SyntaxError:
                    print(f"Line {line_num}: Syntax error in code")
                
                # Check fields
                if name:
                    stats['has_name'] += 1
                if complexity > 0:
                    stats['has_complexity'] += 1
                    complexity_dist[complexity] += 1
                if summary:
                    stats['has_summary'] += 1
                if not summary.strip():
                    stats['empty_summary'] += 1
                
                # Check code length
                code_lines = len([l for l in code.split('\n') if l.strip()])
                if code_lines < 3:
                    stats['short_code'] += 1
                if code_lines > 500:
                    stats['long_code'] += 1
                    
            except json.JSONDecodeError:
                print(f"Line {line_num}: Invalid JSON")
            except Exception as e:
                print(f"Line {line_num}: Error - {e}")
    
    # Print results
    print(f"\nTotal examples: {stats['total']}")
    print(f"Valid Python syntax: {stats['valid_syntax']} ({stats['valid_syntax']/stats['total']*100:.1f}%)")
    print(f"Has function name: {stats['has_name']} ({stats['has_name']/stats['total']*100:.1f}%)")
    print(f"Has complexity: {stats['has_complexity']} ({stats['has_complexity']/stats['total']*100:.1f}%)")
    print(f"Has summary: {stats['has_summary']} ({stats['has_summary']/stats['total']*100:.1f}%)")
    print(f"\nQuality issues:")
    print(f"  Empty summaries: {stats['empty_summary']}")
    print(f"  Very short code (<3 lines): {stats['short_code']}")
    print(f"  Very long code (>500 lines): {stats['long_code']}")
    
    print(f"\nComplexity distribution (top 10):")
    for complexity, count in complexity_dist.most_common(10):
        print(f"  Complexity {complexity}: {count} examples")
    
    print("="*60)
    
    # Validation checks
    passed = True
    if stats['valid_syntax'] < stats['total'] * 0.99:
        print("❌ FAIL: More than 1% of examples have syntax errors")
        passed = False
    if stats['empty_summary'] > 0:
        print("❌ FAIL: Found examples with empty summaries")
        passed = False
    if stats['short_code'] > stats['total'] * 0.01:
        print("❌ FAIL: More than 1% of examples have very short code")
        passed = False
    
    if passed:
        print("✅ PASS: All validation checks passed")
    
    return passed

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python verify_preprocessing.py <file.jsonl>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    passed = verify_preprocessing(file_path)
    sys.exit(0 if passed else 1)
