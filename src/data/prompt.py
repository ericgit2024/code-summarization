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

    Args:
        structural_prompt: The structural prompt of the query code (output of construct_structural_prompt).
        query_code: The code to be summarized.
        retrieved_codes: A list of retrieved similar code snippets.
        retrieved_docstrings: A list of docstrings for the retrieved codes.
        instruction: Optional instruction to guide the LLM.
        repo_context: Optional string containing repository-level context (callers/callees).

    Returns:
        A combined prompt string.
    """
    prompt = ""

    if instruction:
        prompt += f"Instruction: {instruction}\n\n"

    if repo_context:
        prompt += f"Repository Context:\n{repo_context}\n\n"

    prompt += f"Structural Context:\n{structural_prompt}\n\n"

    for i in range(len(retrieved_codes)):
        prompt += f"Retrieved Code Example {i+1}:\n{retrieved_codes[i]}\n"
        prompt += f"Retrieved Summary {i+1}:\n{retrieved_docstrings[i]}\n\n"

    prompt += f"Code to Summarize:\n{query_code}\n\nSummary:"
    return prompt
