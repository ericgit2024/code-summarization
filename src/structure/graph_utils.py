from py2cfg import CFGBuilder
import graphviz

def visualize_cfg(code):
    """
    Generates a visual Control Flow Graph (CFG) from the given Python code using Graphviz.

    Args:
        code (str): The Python code.

    Returns:
        graphviz.Digraph: The graphviz object representing the CFG.
    """
    try:
        cfg_builder = CFGBuilder()
        cfg = cfg_builder.build_from_src("input_code", code)
        
        dot = graphviz.Digraph(comment='Control Flow Graph')
        dot.attr(rankdir='TB')
        
        all_blocks = []
        for func_cfg in cfg.functioncfgs.values():
            all_blocks.extend(list(func_cfg.own_blocks()))

        for block in all_blocks:
            # Create label for the block
            label = f"Block {block.id}\\n"
            # Limit statements to avoid huge nodes
            statements = [str(s).strip() for s in block.statements]
            if len(statements) > 5:
                label += "\\n".join(statements[:5]) + "\\n..."
            else:
                label += "\\n".join(statements)
                
            dot.node(str(block.id), label, shape='box')
            
            for exit in block.exits:
                dot.edge(str(block.id), str(exit.target.id))
                
        return dot
    except Exception as e:
        # Return a simple error graph or None
        err_dot = graphviz.Digraph()
        err_dot.node('error', f"Error generating CFG: {e}")
        return err_dot

def get_cfg(code):
    """
    Generates a Control Flow Graph (CFG) from the given Python code.

    Args:
        code (str): The Python code to generate the CFG from.

    Returns:
        str: A textual representation of the CFG.
    """
    try:
        cfg_builder = CFGBuilder()
        cfg = cfg_builder.build_from_src("input_code", code)

        cfg_text = []
        all_blocks = []
        for func_cfg in cfg.functioncfgs.values():
            all_blocks.extend(list(func_cfg.own_blocks()))

        for block in all_blocks:
            cfg_text.append(f"Block {block.id}:")
            for statement in block.statements:
                cfg_text.append(f"  {statement}")

            if block.exits:
                exits = ", ".join([str(e.target.id) for e in block.exits])
                cfg_text.append(f"  Exits to: {exits}")

        return "\n".join(cfg_text)

    except Exception as e:
        return f"Error generating CFG: {e}"

def get_pdg(code):
    """
    Generates a Program Dependence Graph (PDG) from the given Python code.
    Note: This is a placeholder implementation as py2cfg focuses on CFG.
    Extracting a full PDG usually requires more complex static analysis
    (data dependence + control dependence).

    Args:
        code (str): The Python code.

    Returns:
        str: A textual representation of the PDG (placeholder).
    """
    # For now, we will return the CFG as a proxy or a placeholder
    # Ideally, one would compute data flow analysis here.
    return "PDG extraction not fully implemented. Using CFG."
