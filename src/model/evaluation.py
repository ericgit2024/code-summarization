import evaluate

class CodeEvaluator:
    def __init__(self):
        self.rouge = evaluate.load('rouge')
        self.bleu = evaluate.load('bleu')
        self.meteor = evaluate.load('meteor')

    def compute_metrics(self, predictions, references):
        """
        Computes ROUGE, BLEU, and METEOR scores.

        Args:
            predictions (list of str): The generated summaries.
            references (list of str): The ground truth summaries.

        Returns:
            dict: A dictionary containing the scores.
        """
        if not isinstance(predictions, list) or not isinstance(references, list):
            raise TypeError("Predictions and references must be lists of strings.")

        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length.")

        rouge_results = self.rouge.compute(predictions=predictions, references=references)
        bleu_results = self.bleu.compute(predictions=predictions, references=references)
        meteor_results = self.meteor.compute(predictions=predictions, references=references)

        return {
            "rouge": rouge_results,
            "bleu": bleu_results,
            "meteor": meteor_results
        }

if __name__ == '__main__':
    # Example usage
    evaluator = CodeEvaluator()
    preds = ["hello there", "general kenobi"]
    refs = ["hello there", "general kenobi"]

    scores = evaluator.compute_metrics(preds, refs)
    print(scores)
