import evaluate
from sentence_transformers import SentenceTransformer, util
import logging
from src.structure.graph_utils import get_call_graph, get_cfg
from src.structure.ast_analyzer import ASTAnalyzer

logger = logging.getLogger(__name__)

def calculate_structural_accuracy(code, generated_summary, reference_context=None):
    """
    Calculates a Structural Accuracy Score (SAS) based on:
    1. Dependency Coverage: Does the summary mention the called functions?
    2. Control Flow Awareness: Does it mention loop/branch logic if prominent?

    Returns a score between 0 and 1.
    """
    score = 0.0
    weights = {'dependency': 0.6, 'control_flow': 0.4}

    # 1. Dependency Analysis
    try:
        analyzer = ASTAnalyzer(code)
        analysis = analyzer.analyze()

        # Flatten dependencies from all functions
        dependencies = set()
        for func_meta in analysis['functions'].values():
            for call in func_meta.get('calls', []):
                dependencies.add(call['name'])

        if not dependencies:
            dep_score = 1.0 # No dependencies to miss
        else:
            hits = sum(1 for dep in dependencies if dep.split('.')[-1] in generated_summary)
            dep_score = hits / len(dependencies)

        score += dep_score * weights['dependency']

    except Exception as e:
        logger.warning(f"SAS Dependency Analysis failed: {e}")
        score += 0.5 * weights['dependency'] # Fallback

    # 2. Control Flow Analysis
    try:
        cfg = get_cfg(code)
        # Naive check: if code has loops/branches, summary should mention keywords
        has_loop = 'for ' in code or 'while ' in code
        has_branch = 'if ' in code

        keywords = ['loop', 'iterate', 'check', 'condition', 'if', 'when', 'case']
        summary_lower = generated_summary.lower()

        cf_score = 1.0
        if has_loop or has_branch:
            hit = any(kw in summary_lower for kw in keywords)
            cf_score = 1.0 if hit else 0.5

        score += cf_score * weights['control_flow']

    except Exception as e:
        logger.warning(f"SAS CF Analysis failed: {e}")
        score += 0.5 * weights['control_flow']

    return score

def compute_metrics(predictions, references, code_snippets=None):
    """
    Computes BLEU, ROUGE, METEOR, Semantic Similarity, and optionally Structural Accuracy scores.
    
    Args:
        predictions: List of generated summaries
        references: List of reference summaries
        code_snippets: Optional list of source code snippets for structural analysis
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

    # Structural Accuracy Score (if code snippets provided)
    structural_scores = []
    if code_snippets is not None:
        if len(code_snippets) != len(predictions):
            logger.warning(f"Code snippets count ({len(code_snippets)}) doesn't match predictions count ({len(predictions)})")
        else:
            for code, prediction in zip(code_snippets, predictions):
                sas = calculate_structural_accuracy(code, prediction)
                structural_scores.append(sas)
    
    result = {
        "bleu": bleu_score["bleu"],
        "rouge1": rouge_score_result['rouge1'],
        "rouge2": rouge_score_result['rouge2'],
        "rougeL": rouge_score_result['rougeL'],
        "meteor": meteor_score_result["meteor"],
        "semantic_similarity": semantic_score
    }
    
    # Add structural accuracy if calculated
    if structural_scores:
        result["structural_accuracy"] = sum(structural_scores) / len(structural_scores)
    
    return result
