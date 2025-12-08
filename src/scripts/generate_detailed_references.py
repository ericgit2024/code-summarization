"""
Script to generate detailed reference summaries for validation set.

This script helps create reference summaries that match the detailed output style
of your model, which will improve BLEU/ROUGE/METEOR scores meaningfully.

Usage:
    python -m src.scripts.generate_detailed_references --mode [auto|manual]
    
Modes:
    - auto: Use your trained model to generate initial summaries
    - manual: Provide template for manual creation
"""

import json
import argparse
from pathlib import Path
from src.model.inference import InferencePipeline
from src.data.dataset import load_and_process_dataset


def generate_with_model(dataset, model_path, output_file):
    """
    Generate detailed summaries using your trained model.
    These can then be manually reviewed and corrected.
    """
    print("Loading inference pipeline...")
    pipeline = InferencePipeline(model_path=model_path)
    
    results = []
    
    for idx, example in enumerate(dataset):
        print(f"\nProcessing example {idx + 1}/{len(dataset)}")
        print(f"Code preview: {example['code'][:100]}...")
        
        # Generate summary using your model
        summary = pipeline.generate_from_code(
            code=example['code'],
            use_agent=True,  # Use Smart Agent mode for better quality
            max_attempts=5
        )
        
        results.append({
            "code": example['code'],
            "original_summary": example.get('summary', ''),
            "generated_detailed_summary": summary,
            "manually_verified": False,  # Flag for manual review
            "notes": ""  # For reviewer notes
        })
        
        print(f"Generated summary: {summary[:150]}...")
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Generated {len(results)} detailed summaries")
    print(f"📁 Saved to: {output_file}")
    print("\n⚠️ IMPORTANT: Manually review and correct these summaries before using as references!")


def create_manual_template(dataset, output_file):
    """
    Create a template for manually writing detailed summaries.
    """
    template = []
    
    for idx, example in enumerate(dataset):
        template.append({
            "code": example['code'],
            "original_summary": example.get('summary', ''),
            "detailed_summary_template": {
                "purpose": "TODO: What does this function do?",
                "inputs": "TODO: What are the inputs and their types?",
                "outputs": "TODO: What does it return?",
                "dependencies": "TODO: List functions called (e.g., calls validate(), transform())",
                "logic": "TODO: Step-by-step explanation of internal logic",
                "control_flow": "TODO: Mention loops, conditionals, error handling"
            },
            "final_summary": "TODO: Write a 3-5 sentence detailed summary combining above elements"
        })
    
    with open(output_file, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"\n✅ Created template for {len(template)} examples")
    print(f"📁 Saved to: {output_file}")
    print("\n📝 Fill in the 'final_summary' field for each example")


def convert_to_validation_set(reviewed_file, output_file):
    """
    Convert manually reviewed summaries into final validation set.
    """
    with open(reviewed_file, 'r') as f:
        data = json.load(f)
    
    validation_set = []
    
    for item in data:
        if item.get('manually_verified', False):
            validation_set.append({
                "code": item['code'],
                "summary": item.get('generated_detailed_summary') or item.get('final_summary')
            })
    
    with open(output_file, 'w') as f:
        for item in validation_set:
            f.write(json.dumps(item) + '\n')
    
    print(f"\n✅ Created validation set with {len(validation_set)} verified examples")
    print(f"📁 Saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate detailed reference summaries")
    parser.add_argument('--mode', choices=['auto', 'manual', 'convert'], required=True,
                       help='Generation mode: auto (use model), manual (template), convert (finalize)')
    parser.add_argument('--model_path', default='gemma_lora_finetuned',
                       help='Path to trained model (for auto mode)')
    parser.add_argument('--dataset', default='custom', choices=['custom', 'codexglue'],
                       help='Dataset to use')
    parser.add_argument('--split', default='validation',
                       help='Dataset split to process')
    parser.add_argument('--output', default='detailed_references.json',
                       help='Output file path')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of examples to process')
    
    args = parser.parse_args()
    
    if args.mode == 'convert':
        # Convert reviewed file to final validation set
        convert_to_validation_set(args.output, 'detailed_validation_set.jsonl')
        return
    
    # Load dataset
    print(f"Loading {args.dataset} dataset, split: {args.split}")
    dataset = load_and_process_dataset(split=args.split, dataset_name=args.dataset)
    
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    
    print(f"Processing {len(dataset)} examples")
    
    if args.mode == 'auto':
        generate_with_model(dataset, args.model_path, args.output)
    elif args.mode == 'manual':
        create_manual_template(dataset, args.output)


if __name__ == "__main__":
    main()
