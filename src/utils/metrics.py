import evaluate
from sentence_transformers import SentenceTransformer, util
import logging
from src.structure.graph_utils import get_call_graph, get_cfg
from src.structure.ast_analyzer import ASTAnalyzer
import collections
import torch

# New metrics imports
try:
    from bert_score import score as bert_score_func
except ImportError:
    bert_score_func = None

try:
    from moverscore_v2 import get_idf_dict, word_mover_score
except ImportError:
    word_mover_score = None

try:
    from codebleu import calc_codebleu
except ImportError:
    calc_codebleu = None

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

def compute_metrics(predictions, references, code_snippets=None, full_predictions=None):
    """
    Computes BLEU, ROUGE, METEOR, Semantic Similarity, Structural Accuracy,
    BERTScore, MoverScore, BLEURT, and CodeBLEU.
    
    Args:
        predictions: List of generated summaries (cleaned/shortened for text metrics)
        references: List of reference summaries
        code_snippets: Optional list of source code snippets for structural analysis and CodeBLEU
        full_predictions: Optional list of full generated summaries (for structural analysis)
                          If not provided, uses 'predictions'.
    """
    # 1. Standard Metrics (BLEU, ROUGE, METEOR)
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")

    # BLEU expects a list of references for each prediction
    formatted_references = [[ref] for ref in references]

    bleu_score = bleu_metric.compute(predictions=predictions, references=formatted_references)
    rouge_score_result = rouge_metric.compute(predictions=predictions, references=references)
    meteor_score_result = meteor_metric.compute(predictions=predictions, references=references)

    # 2. Semantic Similarity
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings1 = model.encode(predictions, convert_to_tensor=True)
    embeddings2 = model.encode(references, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    semantic_score = cosine_scores.diagonal().mean().item()

    # 3. BERTScore
    # We use bert_score directly or via evaluate. Evaluate wraps bert_score.
    # We use distilbert-base-uncased for speed and to avoid downloading large models if possible,
    # or let it default (roberta-large is default but heavy).
    # Using 'bert-base-uncased' is a good balance.
    try:
        bertscore = evaluate.load("bertscore")
        bert_results = bertscore.compute(predictions=predictions, references=references, lang="en", model_type="distilbert-base-uncased")
        # Average the F1 scores
        bert_score_f1 = sum(bert_results['f1']) / len(bert_results['f1'])
    except Exception as e:
        logger.warning(f"BERTScore failed: {e}")
        bert_score_f1 = 0.0

    # 4. MoverScore
    try:
        if word_mover_score:
            # Moverscore requires IDF dictionaries
            # We compute IDF based on references and predictions combined or just references?
            # Standard usage:
            idf_dict_hyp = get_idf_dict(predictions)
            idf_dict_ref = get_idf_dict(references)
            ms_scores = word_mover_score(references, predictions, idf_dict_ref, idf_dict_hyp, \
                                      stop_words=[], n_gram=1, remove_subwords=True)
            mover_score_val = sum(ms_scores) / len(ms_scores)
        else:
            mover_score_val = 0.0
    except Exception as e:
        logger.warning(f"MoverScore failed: {e}")
        mover_score_val = 0.0

    # 5. BLEURT
    try:
        # BLEURT requires a checkpoint. Evaluate handles it.
        # Use 'bleurt-tiny' or 'bleurt-base-128' if possible.
        # Evaluate defaults to 'bleurt-base-128' usually.
        bleurt = evaluate.load("bleurt", config_name="bleurt-base-128")
        bleurt_results = bleurt.compute(predictions=predictions, references=references)
        bleurt_score_val = sum(bleurt_results['scores']) / len(bleurt_results['scores'])
    except Exception as e:
        logger.warning(f"BLEURT failed: {e}")
        bleurt_score_val = 0.0

    # 6. CodeBLEU
    # Requires source code.
    codebleu_score_val = 0.0
    if calc_codebleu and code_snippets:
        try:
            # CodeBLEU expects: references (list of lists), predictions (list), lang, weights, tokenizer
            # references in codebleu are [[ref1], [ref2]] if each has 1 ref? No, it expects [[ref1_a, ref2_a, ...], [ref1_b, ref2_b, ...]]?
            # Actually codebleu usage from PyPI package:
            # calc_codebleu(references, predictions, lang, ...)
            # where references is a list of lists of references.
            # predictions is a list of predictions.
            # Wait, if I have N samples.
            # references = [[ref1], [ref2], ..., [refN]]
            # predictions = [pred1, pred2, ..., predN]

            # NOTE: CodeBLEU was designed for code synthesis evaluation (Code vs Code).
            # Here we are evaluating SUMMARY (Text) vs REFERENCE (Text).
            # Using CodeBLEU for text summary evaluation is unusual but requested.
            # HOWEVER, CodeBLEU components (AST match, Dataflow match) rely on the PREDICTION being parseable code.
            # If our prediction is natural language summary, AST/Dataflow match will FAIL or be zero.
            # CodeBLEU is likely inappropriate for Summarization unless the user meant "CodeBLEU" as "Code-aware BLEU" for summarization?
            # Or maybe the user thinks we are generating code? The prompt says "our generated output is paraphrased".

            # If the user specifically requested CodeBLEU for summarization, they might be mistaken about its utility or I might be misunderstanding.
            # But I should provide it. However, if predictions are NL, parser will fail.
            # I will try to run it. If it fails or returns 0, so be it.
            # But wait, CodeBLEU is defined as Weighted average of:
            # 1. N-gram match (standard BLEU)
            # 2. Weighted N-gram match
            # 3. AST match
            # 4. Dataflow match
            # Since predictions are NL, 3 and 4 will likely be 0. 1 and 2 will work.

            # IMPORTANT: The user said "Use these metrics to calculate the benchmark as our generated output is paraphrased".
            # This implies evaluating the text quality.
            # CodeBLEU is generally for Code Generation.
            # But I will implement it as requested.

            formatted_refs_codebleu = [[ref] for ref in references]

            # We assume language is python as per repo.
            # But since predictions are text, we cannot parse them as python.
            # Pass lang="en" or similar? No, CodeBLEU supports programming languages.
            # If I pass lang="python", it will try to parse predictions.
            # It will likely just log errors and return 0 for AST/DF scores.

            result_codebleu = calc_codebleu(formatted_refs_codebleu, predictions, lang="python")
            codebleu_score_val = result_codebleu['codebleu']
        except Exception as e:
            logger.warning(f"CodeBLEU failed: {e}")
            codebleu_score_val = 0.0

    # 7. Structural Accuracy Score
    structural_scores = []
    if code_snippets is not None:
        sas_predictions = full_predictions if full_predictions else predictions
        if len(code_snippets) != len(sas_predictions):
            logger.warning(f"Code snippets count ({len(code_snippets)}) doesn't match predictions count ({len(sas_predictions)})")
        else:
            for code, prediction in zip(code_snippets, sas_predictions):
                sas = calculate_structural_accuracy(code, prediction)
                structural_scores.append(sas)
    
    result = {
        "bleu": bleu_score["bleu"],
        "rouge1": rouge_score_result['rouge1'],
        "rouge2": rouge_score_result['rouge2'],
        "rougeL": rouge_score_result['rougeL'],
        "meteor": meteor_score_result["meteor"],
        "semantic_similarity": semantic_score,
        "bert_score_f1": bert_score_f1,
        "mover_score": mover_score_val,
        "bleurt": bleurt_score_val,
        "codebleu": codebleu_score_val
    }
    
    if structural_scores:
        result["structural_accuracy"] = sum(structural_scores) / len(structural_scores)
    
    return result
