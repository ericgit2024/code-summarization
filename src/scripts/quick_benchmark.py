from src.model.inference import InferencePipeline
from src.data.dataset import load_and_process_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import numpy as np
from tqdm import tqdm
import argparse

def calculate_metrics(predictions, references):
    # BLEU
    bleu_scores = []
    smooth = SmoothingFunction().method1
    for pred, ref in zip(predictions, references):
        # NLTK expects tokenized list of strings
        ref_tokens = ref.split()
        pred_tokens = pred.split()
        score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth)
        bleu_scores.append(score)
    avg_bleu = np.mean(bleu_scores)

    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge_scores.append(scores['rougeL'].fmeasure)
    avg_rouge = np.mean(rouge_scores)

    return avg_bleu, avg_rouge

def run_quick_benchmark(num_samples=5, model_type="codet5"):
    print(f"Loading dataset...")
    dataset = load_and_process_dataset(split="validation")
    limit = min(num_samples, len(dataset))
    subset = dataset.select(range(limit))

    print(f"Initializing Pipeline ({model_type})...")
    pipeline = InferencePipeline(model_type=model_type)

    predictions = []
    references = []

    print(f"Generating...")
    for item in tqdm(subset):
        code = item['code']
        ref = item['summary']
        try:
            pred = pipeline.summarize(code)
        except Exception as e:
            print(f"Error: {e}")
            pred = ""
        predictions.append(pred)
        references.append(ref)

    print("Calculating metrics...")
    bleu, rouge = calculate_metrics(predictions, references)

    print("\n=== Quick Benchmark Results ===")
    print(f"Model: {model_type}")
    print(f"Samples: {limit}")
    print(f"BLEU: {bleu:.4f}")
    print(f"ROUGE-L: {rouge:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="codet5")
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()
    run_quick_benchmark(args.num_samples, args.model_type)
