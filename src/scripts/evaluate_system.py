from src.model.inference import InferencePipeline
from src.data.dataset import load_and_process_dataset
from src.utils.metrics import compute_metrics
from tqdm import tqdm
import json
import random
import argparse

def run_comprehensive_evaluation(num_samples=10, model_type="gemma"):
    print("Loading dataset...")
    dataset = load_and_process_dataset(split="validation")

    # Ensure we have enough samples
    limit = min(num_samples, len(dataset))
    subset = dataset.select(range(limit))

    print(f"Initializing Pipeline with model: {model_type}...")
    pipeline = InferencePipeline(model_type=model_type)

    results = []

    print(f"Evaluating {limit} samples...")
    for item in tqdm(subset):
        code = item['code']
        # Treat 'summary' as the ground truth
        reference = item['summary']

        # 1. Generate Summary
        generated = pipeline.summarize(code)

        # 2. Compute All Metrics (including Structural Accuracy)
        all_metrics = compute_metrics([generated], [reference], code_snippets=[code])

        # 3. Human Eval Placeholder (Likert)
        # In a real scenario, this would pause or log for human review.
        # We will simulate a score based on SAS for automation demonstration.
        sas = all_metrics.get('structural_accuracy', 0.5)
        likert_sim = min(5, max(1, int(sas * 5) + 1))

        results.append({
            'code_snippet': code[:50] + "...",
            'generated': generated,
            'reference': reference,
            'bleu': all_metrics.get('bleu', 0),
            'rouge1': all_metrics.get('rouge1', 0),
            'rouge2': all_metrics.get('rouge2', 0),
            'rougeL': all_metrics.get('rougeL', 0),
            'meteor': all_metrics.get('meteor', 0),
            'semantic_similarity': all_metrics.get('semantic_similarity', 0),
            'structural_accuracy': sas,
            'simulated_human_score': likert_sim
        })


    # Aggregation
    avg_structural_accuracy = sum(r['structural_accuracy'] for r in results) / len(results)
    avg_bleu = sum(r['bleu'] for r in results) / len(results)
    avg_rougeL = sum(r['rougeL'] for r in results) / len(results)
    avg_meteor = sum(r['meteor'] for r in results) / len(results)
    avg_semantic = sum(r['semantic_similarity'] for r in results) / len(results)

    print("\n=== Comprehensive Evaluation Report ===")
    print(f"Average BLEU: {avg_bleu:.4f}")
    print(f"Average ROUGE-L: {avg_rougeL:.4f}")
    print(f"Average METEOR: {avg_meteor:.4f}")
    print(f"Average Semantic Similarity: {avg_semantic:.4f}")
    print(f"Average Structural Accuracy Score (SAS): {avg_structural_accuracy:.4f}")
    print("\nSample Output:")
    print(json.dumps(results[0], indent=2))

    # Save detailed report
    with open("eval_report.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nDetailed report saved to eval_report.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="gemma", help="Model type to use: 'gemma' or 'codet5'")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to evaluate")
    args = parser.parse_args()

    run_comprehensive_evaluation(num_samples=args.num_samples, model_type=args.model_type)
