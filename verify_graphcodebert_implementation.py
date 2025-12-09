"""
Test script to verify GraphCodeBERT implementation without downloading models.
This checks that all modules are importable and have correct structure.
"""

import sys
import importlib.util

def test_module_exists(module_path):
    """Test if a module can be imported."""
    try:
        spec = importlib.util.find_spec(module_path)
        if spec is None:
            return False, f"Module {module_path} not found"
        return True, f"✓ {module_path}"
    except Exception as e:
        return False, f"✗ {module_path}: {e}"

def main():
    print("="*60)
    print("GraphCodeBERT Implementation Verification")
    print("="*60)
    
    modules_to_test = [
        "src.model.graphcodebert_loader",
        "src.model.graphcodebert_inference",
        "src.model.train_graphcodebert",
        "src.scripts.evaluate_graphcodebert",
        "src.scripts.compare_models",
    ]
    
    print("\n1. Testing module imports...")
    all_passed = True
    
    for module in modules_to_test:
        success, message = test_module_exists(module)
        print(f"   {message}")
        if not success:
            all_passed = False
    
    # Test that key functions exist
    print("\n2. Testing key functions...")
    
    try:
        from src.model.graphcodebert_loader import load_graphcodebert
        print("   ✓ load_graphcodebert function found")
    except ImportError as e:
        print(f"   ✗ load_graphcodebert: {e}")
        all_passed = False
    
    try:
        from src.model.graphcodebert_inference import GraphCodeBERTInference
        print("   ✓ GraphCodeBERTInference class found")
    except ImportError as e:
        print(f"   ✗ GraphCodeBERTInference: {e}")
        all_passed = False
    
    try:
        from src.scripts.evaluate_graphcodebert import evaluate_graphcodebert
        print("   ✓ evaluate_graphcodebert function found")
    except ImportError as e:
        print(f"   ✗ evaluate_graphcodebert: {e}")
        all_passed = False
    
    try:
        from src.scripts.compare_models import compare_models
        print("   ✓ compare_models function found")
    except ImportError as e:
        print(f"   ✗ compare_models: {e}")
        all_passed = False
    
    # Check that required files exist
    print("\n3. Checking documentation files...")
    
    import os
    
    files_to_check = [
        "README.md",
        "GRAPHCODEBERT_BASELINE.md",
    ]
    
    for file in files_to_check:
        if os.path.exists(file):
            print(f"   ✓ {file} exists")
        else:
            print(f"   ✗ {file} not found")
            all_passed = False
    
    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("\nYou can now:")
        print("  1. Evaluate zero-shot GraphCodeBERT:")
        print("     python -m src.scripts.evaluate_graphcodebert --mode zeroshot")
        print("\n  2. Train GraphCodeBERT (optional):")
        print("     python -m src.model.train_graphcodebert --epochs 1 --limit 50")
        print("\n  3. Generate comparison report:")
        print("     python -m src.scripts.compare_models")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please review the errors above.")
    print("="*60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
