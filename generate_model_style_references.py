"""
Generate validation references using your model's actual output style.

This script:
1. Generates summaries using your model (Agent mode)
2. Saves them for manual review
3. After review, you can use them as the validation set

Usage:
    python generate_model_style_references.py
"""

import json
from src.model.inference import InferencePipeline

# Load the validation examples
validation_examples = []
with open('detailed_validation_set.jsonl', 'r') as f:
    for line in f:
        validation_examples.append(json.loads(line))

print(f"Loaded {len(validation_examples)} validation examples")
print("\nGenerating summaries using your model's style...\n")

# Initialize pipeline
pipeline = InferencePipeline(model_dir='gemma_lora_finetuned')

# Generate summaries
model_style_references = []

for idx, example in enumerate(validation_examples):
    print(f"{'='*60}")
    print(f"Example {idx + 1}/{len(validation_examples)}")
    print(f"{'='*60}")
    print(f"Code preview: {example['code'][:80]}...")
    
    # Generate with Agent mode (better quality)
    try:
        generated = pipeline.summarize_with_agent(code=example['code'])
    except Exception as e:
        print(f"Agent mode failed: {e}, trying normal mode...")
        generated = pipeline.summarize(code=example['code'])
    
    print(f"\nGenerated summary:\n{generated}\n")
    
    model_style_references.append({
        'code': example['code'],
        'summary': generated,
        'original_reference': example['summary'],
        'manually_verified': False,  # YOU NEED TO VERIFY THIS
        'notes': ''  # Add notes if you edit it
    })

# Save for manual review
output_file = 'model_style_references_for_review.json'
with open(output_file, 'w') as f:
    json.dump(model_style_references, f, indent=2)

print(f"\n{'='*60}")
print(f"✅ Generated {len(model_style_references)} summaries")
print(f"📁 Saved to: {output_file}")
print(f"\n⚠️  IMPORTANT: Manually review and verify these summaries!")
print(f"{'='*60}")
print("\nNext steps:")
print("1. Open 'model_style_references_for_review.json'")
print("2. Review each 'summary' field")
print("3. Fix any errors or hallucinations")
print("4. Set 'manually_verified': true for verified entries")
print("5. Run: python finalize_model_style_references.py")
