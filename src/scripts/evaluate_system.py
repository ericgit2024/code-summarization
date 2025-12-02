from src.model.inference import InferencePipeline
from src.data.dataset import load_and_process_dataset
from src.utils.metrics import compute_metrics
from src.utils.structural_metric import calculate_structural_accuracy
from tqdm import tqdm
import json
import random

def run_comprehensive_evaluation(num_samples=10):
    print("Loading dataset...")
    dataset = load_and_process_dataset(split="validation")

    # Ensure we have enough samples
    limit = min(num_samples, len(dataset))
    subset = dataset.select(range(limit))

    print("Initializing Pipeline...")
    pipeline = InferencePipeline()

    results = []

    print(f"Evaluating {limit} samples...")
    for item in tqdm(subset):
        code = item['code']
        # Treat 'summary' as the ground truth
        reference = item['summary']

        # 1. Generate Summary
        generated = pipeline.summarize(code)

        # 2. Compute Standard Metrics
        nlp_metrics = compute_metrics([generated], [reference])

        # 3. Compute Structural Accuracy Score (SAS)
        sas = calculate_structural_accuracy(code, generated)

        # 4. Human Eval Placeholder (Likert)
        # In a real scenario, this would pause or log for human review.
        # We will simulate a score based on SAS for automation demonstration.
        likert_sim = min(5, max(1, int(sas * 5) + 1))

        results.append({
            'code_snippet': code[:50] + "...",
            'generated': generated,
            'reference': reference,
            'bleu': nlp_metrics.get('bleu', 0),
            'rougeL': nlp_metrics.get('rougeL', 0),
            'sas': sas,
            'simulated_human_score': likert_sim
        })

    # Aggregation
    avg_sas = sum(r['sas'] for r in results) / len(results)
    avg_bleu = sum(r['bleu'] for r in results) / len(results)

    print("\n=== Comprehensive Evaluation Report ===")
    print(f"Average BLEU: {avg_bleu:.4f}")
    print(f"Average Structural Accuracy Score (SAS): {avg_sas:.4f}")
    print("\nSample Output:")
    print(json.dumps(results[0], indent=2))

    # Save detailed report
    with open("eval_report.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nDetailed report saved to eval_report.json")

if __name__ == "__main__":
    run_comprehensive_evaluation()
