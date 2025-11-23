import evaluate
from sentence_transformers import SentenceTransformer, util

def compute_metrics(predictions, references):
    """
    Computes BLEU, ROUGE, METEOR, and Semantic Similarity scores.
    """
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")

    # BLEU expects a list of references for each prediction
    formatted_references = [[ref] for ref in references]

    bleu_score = bleu_metric.compute(predictions=predictions, references=formatted_references)
    rouge_score_result = rouge_metric.compute(predictions=predictions, references=references)
    meteor_score_result = meteor_metric.compute(predictions=predictions, references=references)

    # Semantic Similarity using Sentence Transformers
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings1 = model.encode(predictions, convert_to_tensor=True)
    embeddings2 = model.encode(references, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    # Average diagonal elements (similarity between corresponding pairs)
    semantic_score = cosine_scores.diagonal().mean().item()

    return {
        "bleu": bleu_score["bleu"],
        "rouge1": rouge_score_result['rouge1'],
        "rouge2": rouge_score_result['rouge2'],
        "rougeL": rouge_score_result['rougeL'],
        "meteor": meteor_score_result["meteor"],
        "semantic_similarity": semantic_score
    }
