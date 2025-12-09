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
    block_defs = {}
    block_uses = {}
    
    for block in cfg_blocks:
        bid = block.id
        gen[bid] = set()
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


def get_call_graph_with_files(code, repo_graph=None):
    """
    Generates a Call Graph with file source information.
    
    If repo_graph is provided, looks up the file source for each called function
    and includes it in the output (e.g., "calls foo from utils.py").
    
    Args:
        code: Python code string to analyze
        repo_graph: Optional RepoGraphBuilder instance for file lookup
        
    Returns:
        str: Call graph with file information when available
    """
    try:
        calls = extract_call_graph_edges(code)
        call_graph = []

        for func, func_calls in calls.items():
            if func_calls:
                unique_calls = sorted(list(set(func_calls)))
                
                # If repo_graph is available, enhance with file information
                if repo_graph and hasattr(repo_graph, 'graph'):
                    enhanced_calls = []
                    for called_func in unique_calls:
                        # Try to find the function in the repo graph
                        file_info = _find_function_file(called_func, repo_graph)
                        
                        if file_info:
                            enhanced_calls.append(f"{called_func} (from {file_info})")
                        else:
                            # Function not found in repo (might be built-in or external)
                            enhanced_calls.append(called_func)
                    
                    call_graph.append(f"Function {func} calls: {', '.join(enhanced_calls)}")
                else:
                    # No repo_graph, use basic format
                    call_graph.append(f"Function {func} calls: {', '.join(unique_calls)}")
        
        if not call_graph:
            return "No function calls found."

        return "\n".join(call_graph)

    except Exception as e:
        return f"Error generating Call Graph: {e}"


def _find_function_file(func_name, repo_graph):
    """
    Helper function to find the file source of a function in the repo graph.
    
    Args:
        func_name: Name of the function to find
        repo_graph: RepoGraphBuilder instance
        
    Returns:
        str: Filename (relative path if possible) if found, None otherwise
    """
    import os
    
    # Helper to get display path from node data
    def get_display_path(node_data):
        relative = node_data.get('relativePath')
        if relative:
            return relative.replace(os.sep, '/')
        
        file_path = node_data.get('file_path', '')
        if not file_path:
            return None
            
        # Try to compute relative if root_dir is available in repo_graph
        if hasattr(repo_graph, 'root_dir') and repo_graph.root_dir:
            try:
                return os.path.relpath(file_path, repo_graph.root_dir).replace(os.sep, '/')
            except:
                pass
        
        return os.path.basename(file_path)

    # Try direct match first
    if func_name in repo_graph.graph:
        return get_display_path(repo_graph.graph.nodes[func_name])
    
    # Try partial matches (e.g., "method" might be "Class.method")
    for node in repo_graph.graph.nodes():
        if node.endswith(f".{func_name}") or node == func_name:
            path = get_display_path(repo_graph.graph.nodes[node])
            if path:
                return path
    
    return None


def get_cfg_prompt(code):
    """
    Generates a simplified Control Flow Graph (CFG) prompt for the model.
    Structure:
    Block <ID>:
      <Statements>
      -> Exits to <ID>
    """
    try:
        cfg_builder = CFGBuilder()
        cfg = cfg_builder.build_from_src("input_code", code)

        lines = []
        all_blocks = []
        for func_cfg in cfg.functioncfgs.values():
            all_blocks.extend(list(func_cfg.own_blocks()))
        all_blocks.sort(key=lambda b: b.id)

        for block in all_blocks:
            lines.append(f"Block {block.id}:")
            for stmt in block.statements:
                stmt_code = node_to_simplified_code(stmt)
                lines.append(f"  {stmt_code}")
            
            if block.exits:
                exits = ", ".join([str(e.target.id) for e in block.exits])
                lines.append(f"  -> Exits to: {exits}")
            lines.append("")
            
        return "\n".join(lines)
    except Exception as e:
        return f"Error generating CFG prompt: {e}"

def get_pdg_prompt(code):
    """
    Generates a Program Dependence Graph (PDG) prompt.
    Structure:
    Node <ID>: <Statement>
      <- Controlled by: <ID>
      <- Data from: <ID>(var), ...
    """
    try:
        cfg_builder = CFGBuilder()
        cfg = cfg_builder.build_from_src("input_code", code)
        
        lines = []

        for func_name, func_cfg in cfg.functioncfgs.items():
            lines.append(f"Function: {func_name}")
            
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
            
            # Map for quick lookup
            # dest -> set of sources
            cd_map = {}
            for src, dest in cd_edges:
                if dest not in cd_map: cd_map[dest] = []
                cd_map[dest].append(src)
                
            dd_map = {}
            for src, dest, var in dd_edges:
                if dest not in dd_map: dd_map[dest] = []
                dd_map[dest].append(f"{src}({var})")
            
            for block in blocks:
                # Basic statement representation
                # We combine statements for the block label if multiple
                stmts = [node_to_simplified_code(s) for s in block.statements]
                content = "; ".join(stmts) if stmts else "<Entry/Exit>"
                
                lines.append(f"Node {block.id}: {content}")
                
                # Dependencies
                if block.id in cd_map:
                    refs = ", ".join(map(str, sorted(cd_map[block.id])))
                    lines.append(f"  <- Controlled by: {refs}")
                
                if block.id in dd_map:
                    refs = ", ".join(sorted(dd_map[block.id]))
                    lines.append(f"  <- Data from: {refs}")
            lines.append("")

        return "\n".join(lines)
    except Exception as e:
        return f"Error generating PDG prompt: {e}"

def node_to_simplified_code(node):
    """
    Simplified version of node_to_code for prompts.
    """
    if isinstance(node, ast.AST):
        try:
            return ast.unparse(node).strip()
        except:
            pass
    return str(node).strip()


