from src.structure.ast_utils import get_structural_prompt
from src.structure.graph_utils import get_cfg, get_pdg, get_call_graph

def construct_structural_prompt(code_string):
    """
    Constructs the full structural prompt by fusing AST, CFG, PDG, and Call Graph information.

    Args:
        code_string (str): The python code.

    Returns:
        str: A hierarchical text representation of the code structure.
    """
    ast_prompt = get_structural_prompt(code_string)
    cfg_prompt = get_cfg(code_string)
    pdg_prompt = get_pdg(code_string)
    cg_prompt = get_call_graph(code_string)

    return f"AST:\n{ast_prompt}\n\nCFG:\n{cfg_prompt}\n\nPDG:\n{pdg_prompt}\n\nCall Graph:\n{cg_prompt}"

def construct_prompt(structural_prompt, query_code, retrieved_codes, retrieved_docstrings, instruction=None, repo_context=None):
    """
    Combines structural prompt, retrieved examples, and target code into a single prompt string.
    """
    # Default instruction if none provided
    if not instruction:
        instruction = (
            "You are an expert code summarizer. Your task is to write a concise, natural language summary "
            "of the 'Target Code' provided below. Use the 'Context' information to understand the code's "
            "dependencies and role within the system, but do NOT include raw code or graph data in your summary. "
            "Focus on WHAT the code does and WHY."
        )

    prompt = f"### Instruction\n{instruction}\n\n"

    # Context Section
    prompt += "### Context\n"
    if repo_context:
        prompt += f"**Repository Dependency Context**:\n{repo_context}\n\n"
    
    prompt += f"**Structural Analysis (AST/CFG/Call Graph)**:\n{structural_prompt}\n\n"

    # Few-shot examples (if any)
    if retrieved_codes:
        prompt += "### Examples\n"
        for i in range(len(retrieved_codes)):
            prompt += f"Example {i+1} Code:\n{retrieved_codes[i]}\n"
            prompt += f"Example {i+1} Summary:\n{retrieved_docstrings[i]}\n\n"

    # Target Code
    prompt += f"### Target Code\n{query_code}\n\n"
    
    # Output Indicator
    prompt += "### Natural Language Summary\n"
    return prompt
