import argparse
from src.model.inference import InferencePipeline
from src.data.dataset import load_and_process_dataset
from src.utils.metrics import compute_metrics
from src.utils.text_utils import extract_overview, nlp_to_codexglue
from tqdm import tqdm

def run_benchmark(num_samples=20, use_agent=False):
    """
    Runs the benchmark on a subset of the test dataset.
    """
    print(f"Loading test dataset (Agent: {use_agent})...")
    # Loading validation set as proxy for test if test split is large or unavailable
    dataset = load_and_process_dataset(split="validation")
    dataset = dataset.select(range(min(len(dataset), num_samples)))

    print("Initializing Inference Pipeline...")
    pipeline = InferencePipeline()

    predictions = []
    full_predictions = []
    references = []
    codes = []

    print(f"Generating summaries for {num_samples} examples...")
    for example in tqdm(dataset):
        code = example['code']
        # The dataset uses 'summary' as the key for the reference text
        reference = example.get('docstring', example.get('summary', ''))

        # Generate summary
        if use_agent:
            try:
                summary = pipeline.summarize_with_agent(code=code)
            except Exception as e:
                print(f"Agent failed: {e}")
                summary = pipeline.summarize(code)
        else:
            summary = pipeline.summarize(code)

        # Process summary to match ground truth format for text metrics
        clean_summary = nlp_to_codexglue(summary)

        predictions.append(clean_summary)  # Use cleaned summary for BLEU/ROUGE
        full_predictions.append(summary)   # Keep full summary for SAS (Structural Accuracy)
        references.append(reference)
        codes.append(code)

    print("Computing metrics...")
    metrics = compute_metrics(predictions, references, code_snippets=codes, full_predictions=full_predictions)

    print("\nBenchmark Results:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmark on dataset")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to evaluate")
    parser.add_argument("--use_agent", action="store_true", help="Use Reflective Agent for generation")
    args = parser.parse_args()

    run_benchmark(num_samples=args.num_samples, use_agent=args.use_agent)
