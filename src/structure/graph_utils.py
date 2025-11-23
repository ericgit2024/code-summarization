from py2cfg import CFGBuilder

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
        cfg = cfg_builder.build_from_src(code)

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
