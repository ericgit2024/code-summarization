from src.structure.ast_analyzer import get_ast_prompt
from src.structure.graph_utils import get_cfg_prompt, get_pdg_prompt, get_call_graph, get_call_graph_with_files

def construct_structural_prompt(code_string, repo_graph=None):
    """
    Constructs the full structural prompt by fusing AST, CFG, PDG, and Call Graph information.

    Args:
        code_string (str): The python code.
        repo_graph: Optional RepoGraphBuilder instance for file-aware call graph.

    Returns:
        str: A hierarchical text representation of the code structure.
    """
    ast_prompt = get_ast_prompt(code_string)
    cfg_prompt = get_cfg_prompt(code_string)
    pdg_prompt = get_pdg_prompt(code_string)
    
    # Use enhanced call graph if repo_graph is available
    if repo_graph:
        cg_prompt = get_call_graph_with_files(code_string, repo_graph)
    else:
        cg_prompt = get_call_graph(code_string)

    return f"AST:\n{ast_prompt}\n\nCFG:\n{cfg_prompt}\n\nPDG:\n{pdg_prompt}\n\nCall Graph:\n{cg_prompt}"

def construct_prompt(structural_prompt, query_code, retrieved_codes, retrieved_docstrings, instruction=None, repo_context=None):
    """
    Combines structural prompt, retrieved examples, and target code into a single prompt string.
    """
    # SIMPLIFIED: Match CodeSearchNet docstring format (1-3 sentences, plain language)
    if not instruction:
        instruction = (
            "Generate a concise docstring summary for this code.\n"
            "Write 1-3 sentences explaining what the code does.\n"
            "Do NOT use markdown, bullet points, or structured sections."
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
    
    # Output Indicator - REMOVED to prevent model from copying this label
    # The model should generate the summary directly without seeing "Natural Language Summary:"
    return prompt
