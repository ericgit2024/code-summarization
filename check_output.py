"""
Quick script to see what your model actually generates vs the reference.
"""

import json
from src.model.inference import InferencePipeline

# Load one example
with open('detailed_validation_set.jsonl', 'r') as f:
    example = json.loads(f.readline())

print("="*60)
print("CODE:")
print("="*60)
print(example['code'])
print("\n" + "="*60)
print("REFERENCE SUMMARY:")
print("="*60)
print(example['summary'])

# Generate with your model
pipeline = InferencePipeline(model_dir='gemma_lora_finetuned')
generated = pipeline.summarize(code=example['code'])

print("\n" + "="*60)
print("GENERATED SUMMARY (Normal Mode):")
print("="*60)
print(generated)

print("\n" + "="*60)
print("COMPARISON:")
print("="*60)
print(f"Reference length: {len(example['summary'])} chars")
print(f"Generated length: {len(generated)} chars")
print(f"Reference words: {len(example['summary'].split())}")
print(f"Generated words: {len(generated.split())}")
