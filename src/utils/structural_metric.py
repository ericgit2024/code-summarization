import json
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

if __name__ == "__main__":
    # Test block for immediate verification
    dummy_code = """
def process_data(data):
    if not data:
        return None
    for item in data:
        validate(item)
    return True
"""
    dummy_summary = "The function iterates through the data and calls validate on each item."

    print("Testing Structural Accuracy Score (SAS)...")
    score = calculate_structural_accuracy(dummy_code, dummy_summary)
    print(f"Code:\n{dummy_code}")
    print(f"Summary: {dummy_summary}")
    print(f"Calculated SAS: {score:.4f}")
