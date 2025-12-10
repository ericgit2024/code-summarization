from src.model.inference import InferencePipeline
from src.data.dataset import load_and_process_dataset
from src.utils.metrics import compute_metrics
from src.utils.text_utils import extract_overview
from tqdm import tqdm

def run_benchmark(num_samples=20):
    """
    Runs the benchmark on a subset of the test dataset.
    """
    print("Loading test dataset...")
    # Loading validation set as proxy for test if test split is large or unavailable
    dataset = load_and_process_dataset(split="validation")
    dataset = dataset.select(range(num_samples))

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
        summary = pipeline.summarize(code)

        # Extract Overview for fair comparison with docstrings
        clean_summary = extract_overview(summary)

        predictions.append(clean_summary)
        full_predictions.append(summary)
        references.append(reference)
        codes.append(code)

    print("Computing metrics...")
    metrics = compute_metrics(predictions, references, code_snippets=codes, full_predictions=full_predictions)

    print("\nBenchmark Results:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    run_benchmark()
