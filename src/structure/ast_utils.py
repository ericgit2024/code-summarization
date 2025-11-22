import ast

def get_structural_prompt(code_string):
    """
    Generates a structural prompt from Python code using AST parsing.

    Args:
        code_string: A string containing Python code.

    Returns:
        A string representing the structural prompt.
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
                # Simple assignment
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
