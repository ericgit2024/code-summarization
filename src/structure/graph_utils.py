import networkx as nx
import ast
from py2cfg import CFGBuilder
import graphviz

class DefUseVisitor(ast.NodeVisitor):
    def __init__(self):
        self.defs = set()
        self.uses = set()

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            self.defs.add(node.id)
        elif isinstance(node.ctx, ast.Load):
            self.uses.add(node.id)
    
    def visit_arg(self, node):
        self.defs.add(node.arg)

def get_defs_uses(node):
    visitor = DefUseVisitor()
    visitor.visit(node)
    return visitor.defs, visitor.uses

def compute_control_dependencies(cfg_graph, entry_node):
    """
    Computes control dependencies using post-dominance frontiers.
    """
    try:
        rev_graph = cfg_graph.reverse()
        virtual_exit = "VIRTUAL_EXIT"
        rev_graph.add_node(virtual_exit)
        for node in cfg_graph.nodes():
            if cfg_graph.out_degree(node) == 0:
                rev_graph.add_edge(virtual_exit, node)
        
        dom_frontiers = nx.dominance_frontiers(rev_graph, virtual_exit)
        
        cd_edges = []
        for u, frontier in dom_frontiers.items():
            for v in frontier:
                if u != virtual_exit and v != virtual_exit:
                    cd_edges.append((u, v))
        return cd_edges
    except Exception as e:
        print(f"Error computing control dependencies: {e}")
        return []

def compute_data_dependencies(cfg_blocks, cfg_graph):
    """
    Computes data dependencies using Reaching Definitions analysis.
    """
    gen = {}
    kill = {}
    block_defs = {}
    block_uses = {}
    
    for block in cfg_blocks:
        bid = block.id
        gen[bid] = set()
        kill[bid] = set()
        b_defs = set()
        b_uses = set()
        
        for stmt in block.statements:
            if isinstance(stmt, ast.AST):
                d, u = get_defs_uses(stmt)
                b_defs.update(d)
                b_uses.update(u)
                for var in d:
                    gen[bid].add((var, bid))
        
        block_defs[bid] = b_defs
        block_uses[bid] = b_uses

    all_blocks = [b.id for b in cfg_blocks]
    for bid in all_blocks:
        for var, _ in gen[bid]:
            for other_bid in all_blocks:
                if other_bid != bid:
                    if var in block_defs[other_bid]:
                        kill[bid].add((var, other_bid))

    in_sets = {bid: set() for bid in all_blocks}
    out_sets = {bid: gen[bid].copy() for bid in all_blocks}
    
    changed = True
    while changed:
        changed = False
        for bid in all_blocks:
            new_in = set()
            try:
                preds = list(cfg_graph.predecessors(bid))
                for p in preds:
                    if p in out_sets:
                        new_in.update(out_sets[p])
            except:
                pass 
            
            if new_in != in_sets[bid]:
                in_sets[bid] = new_in
                changed = True
            
            surviving_in = set()
            for var, source_block in in_sets[bid]:
                if var not in block_defs[bid]:
                    surviving_in.add((var, source_block))
            
            new_out = gen[bid].union(surviving_in)
            
            if new_out != out_sets[bid]:
                out_sets[bid] = new_out
                changed = True

    dd_edges = []
    for bid in all_blocks:
        for var in block_uses[bid]:
            for def_var, def_block in in_sets[bid]:
                if var == def_var:
                    dd_edges.append((def_block, bid, var))
            
    return dd_edges

def visualize_cfg(code):
    """
    Generates a visual Control Flow Graph (CFG) from the given Python code using Graphviz.
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
            label = f"Block {block.id}\\n"
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
        err_dot = graphviz.Digraph()
        err_dot.node('error', f"Error generating CFG: {e}")
        return err_dot

def get_cfg(code):
    """
    Generates a Control Flow Graph (CFG) from the given Python code.
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

        return "\\n".join(cfg_text)

    except Exception as e:
        return f"Error generating CFG: {e}"

def get_pdg(code):
    """
    Generates a Program Dependence Graph (PDG) from the given Python code.
    Computes both Control Dependencies and Data Dependencies.
    """
    try:
        cfg_builder = CFGBuilder()
        cfg = cfg_builder.build_from_src("input_code", code)
        
        pdg_output = []

        for func_name, func_cfg in cfg.functioncfgs.items():
            pdg_output.append(f"Function: {func_name}")
            
            blocks = list(func_cfg.own_blocks())
            blocks.sort(key=lambda b: b.id)
            
            if not blocks:
                continue
                
            entry_block_id = blocks[0].id
            
            nx_graph = nx.DiGraph()
            
            for block in blocks:
                nx_graph.add_node(block.id)
                for exit in block.exits:
                    nx_graph.add_edge(block.id, exit.target.id)
            
            cd_edges = compute_control_dependencies(nx_graph, entry_block_id)
            dd_edges = compute_data_dependencies(blocks, nx_graph)
            
            node_dependencies = {b.id: {'control': [], 'data': []} for b in blocks}
            
            for src, dest in cd_edges:
                if dest in node_dependencies:
                    node_dependencies[dest]['control'].append(src)
            
            for src, dest, var in dd_edges:
                if dest in node_dependencies:
                    node_dependencies[dest]['data'].append(f"{src}({var})")
            
            for block in blocks:
                pdg_output.append(f"  Node {block.id}:")
                for stmt in block.statements:
                    pdg_output.append(f"    {str(stmt).strip()}")
                
                deps = node_dependencies[block.id]
                if deps['control']:
                    pdg_output.append(f"    Control Dependent on: {', '.join(map(str, deps['control']))}")
                if deps['data']:
                    pdg_output.append(f"    Data Dependent on: {', '.join(deps['data'])}")
            
            pdg_output.append("")

        return "\\n".join(pdg_output)

    except Exception as e:
        return f"Error generating PDG: {e}"
