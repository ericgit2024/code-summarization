"""
Test script to verify file-level dependency enhancement.

Creates a multi-file test repository and generates a summary to verify
that file sources are mentioned for dependencies.
"""

import os
import sys

# Create test files
test_dir = "test_file_deps"
os.makedirs(test_dir, exist_ok=True)

# File 1: math_utils.py
math_utils_code = '''def lcm(a, b):
    """Calculate the least common multiple of two numbers."""
    from math import gcd
    return abs(a * b) // gcd(a, b)

def factorial(n):
    """Calculate factorial of n."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)
'''

# File 2: calculator.py
calculator_code = '''from math_utils import lcm, factorial

def sum_with_lcm(numbers):
    """
    Calculate sum of numbers and normalize using LCM.
    
    This function computes the sum of a list of numbers and then
    normalizes the result using the least common multiple.
    """
    if not numbers:
        return 0
    
    total = sum(numbers)
    
    # Use LCM for normalization
    if len(numbers) >= 2:
        norm_factor = lcm(numbers[0], numbers[1])
        total = total / norm_factor
    
    return total

def compute_factorial_sum(n):
    """Compute sum of factorials from 1 to n."""
    result = 0
    for i in range(1, n + 1):
        result += factorial(i)
    return result
'''

# Write test files
with open(os.path.join(test_dir, "math_utils.py"), "w") as f:
    f.write(math_utils_code)

with open(os.path.join(test_dir, "calculator.py"), "w") as f:
    f.write(calculator_code)

print("="*60)
print("TEST: File-Level Dependency Enhancement")
print("="*60)
print(f"\nCreated test repository in: {test_dir}/")
print("  - math_utils.py (contains lcm, factorial)")
print("  - calculator.py (contains sum_with_lcm, compute_factorial_sum)")

# Now test the inference pipeline
try:
    from src.model.inference import InferencePipeline
    
    print("\n" + "="*60)
    print("Building repository graph...")
    print("="*60)
    
    pipeline = InferencePipeline(allow_mock=True)
    pipeline.build_repo_graph(test_dir)
    
    print(f"\nGraph contains {len(pipeline.repo_graph.graph.nodes())} nodes:")
    for node in pipeline.repo_graph.graph.nodes():
        print(f"  - {node}")
    
    print("\n" + "="*60)
    print("Generating summary for 'sum_with_lcm'...")
    print("="*60)
    
    summary = pipeline.summarize(function_name="sum_with_lcm")
    
    print("\n" + "="*60)
    print("GENERATED SUMMARY:")
    print("="*60)
    print(summary)
    
    print("\n" + "="*60)
    print("VERIFICATION:")
    print("="*60)
    
    # Check if file mentions are present
    checks = [
        ("lcm() from math_utils.py", "lcm" in summary.lower() and "math_utils" in summary.lower()),
        ("File mention format", "from" in summary.lower() and ".py" in summary.lower()),
    ]
    
    all_passed = True
    for check_name, result in checks:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {check_name}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 SUCCESS: File-level dependencies are mentioned in the summary!")
    else:
        print("\n⚠️  WARNING: Some checks failed. Review the summary above.")
        print("\nExpected format: 'calls lcm() from math_utils.py'")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Cleanup
    print("\n" + "="*60)
    print("Cleaning up test files...")
    print("="*60)
    import shutil
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        print(f"✓ Removed {test_dir}/")
