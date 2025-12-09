from src.model.inference import InferencePipeline
from src.data.dataset import load_and_process_dataset
from src.utils.metrics import compute_metrics
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
    references = []
    codes = []

    print(f"Generating summaries for {num_samples} examples...")
    for example in tqdm(dataset):
        code = example['code']
        # The dataset uses 'summary' as the key for the reference text
        reference = example.get('docstring', example.get('summary', ''))

        # Generate summary
        summary = pipeline.summarize(code)

        # Extract 'Overview' for fair metric comparison
        # The generated summary is structured Markdown. We want to compare the 'Overview'
        # against the reference docstring to improve BLEU/ROUGE scores.
        summary_for_eval = summary
        if "**Overview**" in summary:
            try:
                # Extract text between **Overview** and the next section (usually **Detailed Logic**)
                parts = summary.split("**Overview**")
                if len(parts) > 1:
                    content = parts[1]
                    # Find end of section (next double asterisk or newline sequence)
                    # Often the next header is "**Detailed Logic**"
                    if "**Detailed Logic**" in content:
                        content = content.split("**Detailed Logic**")[0]

                    # Clean up
                    content = content.replace(":", "", 1).strip() # Remove leading colon if present
                    summary_for_eval = content
            except Exception as e:
                print(f"Warning: Failed to extract Overview: {e}")

        predictions.append(summary_for_eval)
        references.append(reference)
        codes.append(code)

    print("Computing metrics...")
    metrics = compute_metrics(predictions, references, code_snippets=codes)

    print("\nBenchmark Results:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    run_benchmark()
