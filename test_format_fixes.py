"""
Test script to verify that generated summaries match the natural language format
of CodeSearchNet dataset references (not structured markdown).
"""

from src.model.inference import InferencePipeline, clean_summary_for_evaluation
import re

def test_post_processing():
    """Test the clean_summary_for_evaluation function"""
    print("="*60)
    print("TEST 1: Post-Processing Function")
    print("="*60)
    
    test_cases = [
        # JSON wrapper
        ('{"docstring": "Check if input has valid values"}', "Check if input has valid values"),
        
        # Markdown headers
        ('**Overview**: Validates input data', "Validates input data"),
        
        # Section markers
        ('Natural Language Overview\nCreate a new product in database', "Create a new product in database"),
        
        # Args/Returns format
        ('Args: name, price\nReturns: Product object', "name, price Product object"),
        
        # Multiple sections
        ('**Overview**: Creates product\n**Detailed Logic**: Validates then saves\n**Dependency Analysis**: Calls validate()', 
         "Creates product Validates then saves Calls validate()"),
    ]
    
    passed = 0
    for i, (input_text, expected_contains) in enumerate(test_cases, 1):
        result = clean_summary_for_evaluation(input_text)
        # Check if key content is preserved (not exact match due to whitespace normalization)
        if expected_contains.lower() in result.lower():
            print(f"✓ Test {i} PASSED")
            passed += 1
        else:
            print(f"✗ Test {i} FAILED")
            print(f"  Input: {input_text[:50]}...")
            print(f"  Expected to contain: {expected_contains}")
            print(f"  Got: {result}")
    
    print(f"\nPost-processing tests: {passed}/{len(test_cases)} passed\n")
    return passed == len(test_cases)

def test_summary_format():
    """Test that generated summaries are natural language, not structured"""
    print("="*60)
    print("TEST 2: Generated Summary Format")
    print("="*60)
    
    # Simple test code
    test_code = '''
def validate_input(data):
    """Check if data is valid"""
    if not data:
        raise ValueError("Data is empty")
    return True
'''
    
    try:
        pipeline = InferencePipeline(allow_mock=True)
        summary = pipeline.summarize(code=test_code)
        
        print(f"Generated summary:\n{summary}\n")
        
        # Check for bad patterns (structured markdown)
        bad_patterns = [
            r'\*\*Overview\*\*',
            r'\*\*Detailed Logic\*\*',
            r'\*\*Dependency Analysis\*\*',
            r'##\s+',
            r'Natural Language Overview',
            r'Target Code Examples',
        ]
        
        issues = []
        for pattern in bad_patterns:
            if re.search(pattern, summary, re.IGNORECASE):
                issues.append(f"Found structured pattern: {pattern}")
        
        # Check for good patterns (natural language)
        good_indicators = [
            len(summary.split()) > 10,  # At least 10 words
            len(summary.split()) < 200,  # Not too verbose
            not summary.startswith('{'),  # Not JSON
            '**' not in summary,  # No markdown bold
        ]
        
        if issues:
            print("✗ FAILED: Found structured formatting:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        elif all(good_indicators):
            print("✓ PASSED: Summary is natural language format")
            return True
        else:
            print("⚠ WARNING: Summary format unclear")
            return False
            
    except Exception as e:
        print(f"✗ FAILED: Error during generation: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("SUMMARY FORMAT VERIFICATION TESTS")
    print("="*60 + "\n")
    
    test1_pass = test_post_processing()
    test2_pass = test_summary_format()
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Post-Processing Tests: {'✓ PASSED' if test1_pass else '✗ FAILED'}")
    print(f"Summary Format Tests: {'✓ PASSED' if test2_pass else '✗ FAILED'}")
    
    if test1_pass and test2_pass:
        print("\n✓ ALL TESTS PASSED - Format fixes are working correctly!")
        print("\nNext step: Run full evaluation with:")
        print("  python src/scripts/evaluate_system.py --num_samples 20")
    else:
        print("\n✗ SOME TESTS FAILED - Review the output above")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
