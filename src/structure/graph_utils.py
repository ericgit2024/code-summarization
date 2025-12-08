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

    def visit_FunctionDef(self, node):
        self.defs.add(node.name)
        # Do not visit the body of the function

    def visit_ClassDef(self, node):
        self.defs.add(node.name)
        # Do not visit the body of the class

def get_defs_uses(node):
    visitor = DefUseVisitor()
    visitor.visit(node)
    return visitor.defs, visitor.uses

def node_to_code(node):
    if isinstance(node, ast.AST):
        try:
            return ast.unparse(node).strip()
        except Exception:
            return ast.dump(node)
    return str(node).strip()

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
                    # u dominates a predecessor of v in R-CFG, but not v itself.
                    # This means v is control dependent on u.
                    # So u is the controller, v is the dependent.
                    cd_edges.append((u, v))
        return cd_edges
    except Exception as e:
        print(f"Error computing control dependencies: {e}")
        return []

def compute_data_dependencies(cfg_blocks, cfg_graph, func_args=None):
    """
    Computes data dependencies using Reaching Definitions analysis.
    """
    gen = {}
    block_defs = {}
    block_uses = {}
    
    # Identify entry block to inject arguments as definitions
    entry_block_id = None
    if cfg_blocks:
        entry_block_id = sorted(cfg_blocks, key=lambda b: b.id)[0].id

    for block in cfg_blocks:
        bid = block.id
        gen[bid] = set()
        b_defs = set()
        b_uses = set()
        
        # Inject function arguments as definitions in the entry block
        if bid == entry_block_id and func_args:
            for arg in func_args:
                b_defs.add(arg)
                gen[bid].add((arg, bid))

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
            
            # Kill logic: remove incoming definitions if the variable is redefined in this block
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
            statements = [node_to_code(s) for s in block.statements]
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
            
            # Format statements with proper indentation
            for statement in block.statements:
                stmt_code = node_to_code(statement)
                # Handle multi-line statements
                if '\n' in stmt_code:
                    lines = stmt_code.split('\n')
                    for line in lines:
                        cfg_text.append(f"  {line}")
                else:
                    cfg_text.append(f"  {stmt_code}")

            # Add exit information
            if block.exits:
                exits = ", ".join([str(e.target.id) for e in block.exits])
                cfg_text.append(f"  → Exits to: {exits}")
            
            # Add blank line between blocks for readability
            cfg_text.append("")

        return "\n".join(cfg_text)

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

        # Pre-process AST to map function names to their arguments
        # This handles top-level functions which py2cfg reliably detects.
        func_args_map = {}
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_args_map[node.name] = [arg.arg for arg in node.args.args]
        except Exception as e:
            # Fallback if AST parsing fails
            pass

        for func_name, func_cfg in cfg.functioncfgs.items():
            pdg_output.append(f"Function: {func_name}")
            pdg_output.append("=" * 60)
            
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
            
            # Retrieve arguments for this function
            func_args = func_args_map.get(func_name, [])

            cd_edges = compute_control_dependencies(nx_graph, entry_block_id)
            dd_edges = compute_data_dependencies(blocks, nx_graph, func_args)
            
            node_dependencies = {b.id: {'control': [], 'data': []} for b in blocks}
            
            for controller, dependent in cd_edges:
                # Correct Logic: "dependent" depends on "controller"
                # Store it as: node_dependencies[dependent]['control'].append(controller)
                if dependent in node_dependencies:
                    node_dependencies[dependent]['control'].append(controller)
            
            for src, dest, var in dd_edges:
                if dest in node_dependencies:
                    node_dependencies[dest]['data'].append(f"{src}({var})")
            
            for block in blocks:
                pdg_output.append(f"\nNode {block.id}:")
                
                # Format statements with proper indentation
                for stmt in block.statements:
                    stmt_code = node_to_code(stmt)
                    # Handle multi-line statements
                    if '\n' in stmt_code:
                        lines = stmt_code.split('\n')
                        for line in lines:
                            pdg_output.append(f"  {line}")
                    else:
                        pdg_output.append(f"  {stmt_code}")
                
                # Add dependencies with clear labels
                deps = node_dependencies[block.id]
                if deps['control']:
                    pdg_output.append(f"  ├─ Control Dependencies: {', '.join(map(str, deps['control']))}")
                if deps['data']:
                    pdg_output.append(f"  └─ Data Dependencies: {', '.join(deps['data'])}")
            
            pdg_output.append("")

        return "\n".join(pdg_output)

    except Exception as e:
        return f"Error generating PDG: {e}"

def extract_call_graph_edges(code):
    """
    Extracts call graph edges from the given Python code.
    Returns a dictionary mapping function names to lists of called function names.
    """
    try:
        tree = ast.parse(code)
        
        class CallVisitor(ast.NodeVisitor):
            def __init__(self):
                self.current_func = "Module"
                self.calls = {} # Map function name to list of called functions

            def visit_FunctionDef(self, node):
                prev_func = self.current_func
                self.current_func = node.name
                if self.current_func not in self.calls:
                    self.calls[self.current_func] = []
                self.generic_visit(node)
                self.current_func = prev_func

            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    called_func = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    called_func = node.func.attr
                else:
                    called_func = "unknown"
                
                if self.current_func not in self.calls:
                    self.calls[self.current_func] = []
                self.calls[self.current_func].append(called_func)
                self.generic_visit(node)

        visitor = CallVisitor()
        visitor.visit(tree)
        return visitor.calls
    except Exception as e:
        print(f"Error extracting call graph edges: {e}")
        return {}

def get_call_graph(code):
    """
    Generates a Call Graph from the given Python code.
    Identifies which functions call which other functions.
    """
    try:
        calls = extract_call_graph_edges(code)
        call_graph = []

        for func, func_calls in calls.items():
            if func_calls:
                unique_calls = sorted(list(set(func_calls)))
                call_graph.append(f"Function {func} calls: {', '.join(unique_calls)}")
        
        if not call_graph:
            return "No function calls found."

        return "\n".join(call_graph)

    except Exception as e:
        return f"Error generating Call Graph: {e}"
