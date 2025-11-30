import os
import argparse
from tqdm import tqdm
from src.model.inference import InferencePipeline
from src.data.dataset import load_and_process_dataset
from src.utils.metrics import compute_metrics
import torch
import gc

def generate_predictions(pipeline, dataset, num_samples):
    predictions = []
    references = []

    # Select subset
    subset = dataset.select(range(num_samples))

    for example in tqdm(subset, desc="Generating summaries"):
        code = example['code']
        # Fallback to 'summary' if 'docstring' is missing
        reference = example.get('docstring', example.get('summary', ''))

        try:
            summary = pipeline.summarize(code=code)
        except Exception as e:
            print(f"Error generating summary: {e}")
            summary = ""

        predictions.append(summary)
        references.append(reference)

    return predictions, references

def run_comparison(model1_dir, model2_dir, num_samples=10, output_file="comparison_results.md"):
    print(f"Loading dataset...")
    try:
        dataset = load_and_process_dataset(split="validation")
    except ValueError:
        print("Validation split not found. Using 'train' split.")
        dataset = load_and_process_dataset(split="train")

    if len(dataset) < num_samples:
        print(f"Warning: Dataset size ({len(dataset)}) is smaller than requested samples ({num_samples}). using full dataset.")
        num_samples = len(dataset)

    results = {}
    samples = {}

    # --- Model 1 ---
    print(f"\nEvaluating Model 1: {model1_dir}...")
    pipeline1 = InferencePipeline(model_dir=model1_dir)
    preds1, refs = generate_predictions(pipeline1, dataset, num_samples)
    metrics1 = compute_metrics(preds1, refs)
    results["Model 1"] = metrics1
    samples["Model 1"] = preds1

    # Cleanup Model 1 to save memory
    del pipeline1
    torch.cuda.empty_cache()
    gc.collect()

    # --- Model 2 ---
    print(f"\nEvaluating Model 2: {model2_dir}...")
    pipeline2 = InferencePipeline(model_dir=model2_dir)
    preds2, _ = generate_predictions(pipeline2, dataset, num_samples)
    metrics2 = compute_metrics(preds2, refs)
    results["Model 2"] = metrics2
    samples["Model 2"] = preds2

    # Cleanup Model 2
    del pipeline2
    torch.cuda.empty_cache()
    gc.collect()

    # --- Reporting ---
    print("\n--- Comparison Results ---")

    # Metrics Table
    header = f"| Metric | Model 1 ({model1_dir}) | Model 2 ({model2_dir}) |"
    sep = "| --- | --- | --- |"
    rows = []
    for key in metrics1.keys():
        rows.append(f"| {key} | {metrics1[key]:.4f} | {metrics2[key]:.4f} |")

    table = "\n".join([header, sep] + rows)
    print(table)

    # Write to file
    with open(output_file, "w") as f:
        f.write(f"# Model Comparison\n\n")
        f.write(f"**Model 1**: {model1_dir}\n")
        f.write(f"**Model 2**: {model2_dir}\n\n")
        f.write("## Metrics\n\n")
        f.write(table + "\n\n")

        f.write("## Sample Outputs\n\n")
        for i in range(min(5, num_samples)):
            f.write(f"### Example {i+1}\n")
            f.write(f"**Reference**:\n{refs[i]}\n\n")
            f.write(f"**Model 1 Output**:\n{samples['Model 1'][i]}\n\n")
            f.write(f"**Model 2 Output**:\n{samples['Model 2'][i]}\n\n")
            f.write("---\n")

    print(f"\nDetailed results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two models on the code summarization task.")
    parser.add_argument("--model1", type=str, default="gemma_lora_finetuned", help="Path to first model (or 'base')")
    parser.add_argument("--model2", type=str, default="base", help="Path to second model (or 'base')")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--output", type=str, default="comparison_results.md", help="Output file for results")

    args = parser.parse_args()

    # Handle 'base' keyword by passing None so InferencePipeline defaults to base
    m1 = args.model1 if args.model1 != "base" else None
    m2 = args.model2 if args.model2 != "base" else None

    run_comparison(m1, m2, args.num_samples, args.output)
