import evaluate
import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class CodeSummaryEvaluator:
    def __init__(self):
        # Load standard metrics
        self.bert_score = evaluate.load("bertscore")
        self.meteor = evaluate.load("meteor")
        self.rouge = evaluate.load("rouge")
        self.bleu = evaluate.load("bleu")

    def evaluate_summary(self, reference, hypothesis):
        """
        Evaluates a single summary against a reference.
        Returns a dictionary of metrics.
        """
        # Prepare inputs (list format required by some libraries)
        refs = [reference]
        hyps = [hypothesis]

        results = {}

        # 1. BLEU
        # Smoothing function for short sequences
        smooth = SmoothingFunction().method1
        # NLTK BLEU for single sentence
        ref_tokens = reference.split()
        hyp_tokens = hypothesis.split()
        results["bleu_4"] = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smooth)

        # 2. ROUGE
        rouge_results = self.rouge.compute(predictions=hyps, references=refs)
        results["rouge1"] = rouge_results["rouge1"]
        results["rouge2"] = rouge_results["rouge2"]
        results["rougeL"] = rouge_results["rougeL"]

        # 3. METEOR
        meteor_results = self.meteor.compute(predictions=hyps, references=refs)
        results["meteor"] = meteor_results["meteor"]

        # 4. BERTScore (Semantic)
        # Using a small model for speed, can be configured
        try:
            bert_results = self.bert_score.compute(predictions=hyps, references=refs, lang="en", model_type="distilbert-base-uncased")
            results["bert_f1"] = np.mean(bert_results["f1"])
        except Exception as e:
            print(f"BERTScore failed: {e}")
            results["bert_f1"] = 0.0

        return results

    def compare_models(self, references, model_predictions):
        """
        Compares multiple models.
        model_predictions: dict {model_name: [list of hypotheses]}
        references: list of reference summaries
        """
        report = {}

        for model_name, hyps in model_predictions.items():
            if len(hyps) != len(references):
                print(f"Warning: {model_name} has {len(hyps)} predictions, expected {len(references)}.")
                continue

            print(f"Evaluating {model_name}...")
            model_scores = {
                "bleu_4": [], "rouge1": [], "rouge2": [], "rougeL": [], "meteor": [], "bert_f1": []
            }

            for ref, hyp in zip(references, hyps):
                scores = self.evaluate_summary(ref, hyp)
                for k, v in scores.items():
                    model_scores[k].append(v)

            # Aggregate
            aggregated = {k: np.mean(v) for k, v in model_scores.items()}
            report[model_name] = aggregated

        return report

    def print_comparison(self, report):
        """
        Pretty prints the comparison report.
        """
        metrics = ["bleu_4", "rouge1", "rouge2", "rougeL", "meteor", "bert_f1"]

        # Header
        header = f"{'Model':<20} | " + " | ".join([f"{m:<10}" for m in metrics])
        print("-" * len(header))
        print(header)
        print("-" * len(header))

        for model, scores in report.items():
            row = f"{model:<20} | " + " | ".join([f"{scores.get(m, 0):.4f}" for m in metrics])
            print(row)
