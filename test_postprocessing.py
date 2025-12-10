"""Quick test to verify the improved post-processing handles dict literals"""
import sys
sys.path.insert(0, '.')

from src.model.inference import clean_summary_for_evaluation

# Test case from user's actual output
test_input = "{'doc': \"Method to set the text of this element.\"} {'body': 'Set the text of this element.\\nAligns with the ``current'' class if it is used on an Element.\\t\\tArguments:\\tthe text\\ttt:\\tThe text', '':"

print("Input:", test_input[:100], "...")
print()

result = clean_summary_for_evaluation(test_input)

print("Output:", result)
print()
print("Length:", len(result.split()), "words")
print("Has dict markers:", '{' in result or '}' in result)
