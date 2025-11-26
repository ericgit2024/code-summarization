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
        
        for block_id, block in cfg.blocks.items():
            # Create label for the block
            label = f"Block {block_id}\\n"
            # Limit statements to avoid huge nodes
            statements = [str(s).strip() for s in block.statements]
            if len(statements) > 5:
                label += "\\n".join(statements[:5]) + "\\n..."
            else:
                label += "\\n".join(statements)
                
            dot.node(str(block_id), label, shape='box')
            
            for exit in block.exits:
                dot.edge(str(block_id), str(exit.target.id))
                
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

        # Convert the CFG graph to a textual representation
        # This is a simplified representation for prompting purposes
        # You might want to refine this based on the output format of py2cfg

        cfg_text = []
        for block_id, block in cfg.blocks.items():
            cfg_text.append(f"Block {block_id}:")
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
