import ast
from py2cfg import CFGBuilder

def get_ast_prompt(code_string):
    """
    Generates a structural prompt from Python code using AST parsing.
    """
    try:
        tree = ast.parse(code_string)
        prompt_parts = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                prompt_parts.append(f"Function: {node.name}")
                for arg in node.args.args:
                    prompt_parts.append(f"  Arg: {arg.arg}")
            elif isinstance(node, ast.ClassDef):
                prompt_parts.append(f"Class: {node.name}")
            elif isinstance(node, ast.Assign):
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                     prompt_parts.append(f"Assignment: {node.targets[0].id}")
            elif isinstance(node, ast.Import):
                 for alias in node.names:
                     prompt_parts.append(f"Import: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                 module = node.module if node.module else ""
                 for alias in node.names:
                     prompt_parts.append(f"Import from {module}: {alias.name}")

        return "\n".join(prompt_parts)
    except SyntaxError as e:
        return f"Error parsing code: {e}"

def get_cfg_prompt(code_string):
    """
    Generates a textual representation of the Control Flow Graph for the given code.
    """
    cfg = CFGBuilder().build_from_src("code", code_string)
    prompt_parts = []
    for block in cfg.own_blocks():
        prompt_parts.append(f"Block {block.id}:")
        source = block.get_source()
        if source:
            prompt_parts.append(f"  Source: {source.strip()}")
        calls = block.get_calls()
        if calls:
            prompt_parts.append(f"  Calls: {calls}")
        for exit_link in block.exits:
            exit_case = exit_link.get_exitcase()
            if exit_case:
                prompt_parts.append(f"  Exit to Block {exit_link.target.id} on condition: {exit_case.strip()}")
            else:
                prompt_parts.append(f"  Exit to Block {exit_link.target.id}")
    return "\n".join(prompt_parts)


def get_combined_structural_prompt(code_string):
    """
    Generates a combined structural prompt from both AST and CFG.
    """
    ast_prompt = get_ast_prompt(code_string)
    cfg_prompt = get_cfg_prompt(code_string)
    return f"AST:\n{ast_prompt}\n\nCFG:\n{cfg_prompt}"

if __name__ == '__main__':
    from .data_loader import load_and_clean_dataset
    dataset = load_and_clean_dataset()
    sample_code = dataset['train'][0]['code']
    structural_prompt = get_combined_structural_prompt(sample_code)
    print("Sample Code:")
    print(sample_code)
    print("\nGenerated Structural Prompt:")
    print(structural_prompt)
