"""
Finalize model-style references after manual review.

This converts the reviewed summaries into the final validation set.

Usage:
    python finalize_model_style_references.py
"""

import json

# Load the reviewed references
with open('model_style_references_for_review.json', 'r') as f:
    reviewed_data = json.load(f)

# Filter only verified entries
verified_entries = [
    entry for entry in reviewed_data 
    if entry.get('manually_verified', False)
]

if len(verified_entries) == 0:
    print("❌ No verified entries found!")
    print("Please review 'model_style_references_for_review.json' and set 'manually_verified': true")
    exit(1)

print(f"Found {len(verified_entries)} verified entries out of {len(reviewed_data)}")

# Create final validation set
final_validation = []
for entry in verified_entries:
    final_validation.append({
        'code': entry['code'],
        'summary': entry['summary']
    })

# Save as JSONL
output_file = 'model_style_validation_set.jsonl'
with open(output_file, 'w') as f:
    for item in final_validation:
        f.write(json.dumps(item) + '\n')

print(f"\n✅ Created validation set with {len(final_validation)} examples")
print(f"📁 Saved to: {output_file}")
print("\nNow run evaluation:")
print(f"  python -m src.scripts.evaluate_detailed_validation --mode agent")
print("\nNote: Update the script to use 'model_style_validation_set.jsonl' instead")
