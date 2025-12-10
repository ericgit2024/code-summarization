import networkx as nx
import ast
from py2cfg import CFGBuilder
import graphviz
import logging

logger = logging.getLogger(__name__)

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
    Returns list of (controller, controlled) edges.
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
                    # In dominance frontiers of reverse graph:
                    # v is in PDF(u) means u depends on v.
                    # v is the controller, u is the controlled.
                    cd_edges.append((v, u))
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
                        file_info = _find_function_file(called_func, repo_graph)
                        if file_info:
                            enhanced_calls.append(f"{called_func} (from {file_info})")
                        else:
                            enhanced_calls.append(called_func)
                    
                    call_graph.append(f"Function {func} calls: {', '.join(enhanced_calls)}")
                else:
                    call_graph.append(f"Function {func} calls: {', '.join(unique_calls)}")
        
        if not call_graph:
            return "No function calls found."

        return "\n".join(call_graph)

    except Exception as e:
        return f"Error generating Call Graph: {e}"


def _find_function_file(func_name, repo_graph):
    import os
    def get_display_path(node_data):
        relative = node_data.get('relativePath')
        if relative:
            return relative.replace(os.sep, '/')
        file_path = node_data.get('file_path', '')
        if not file_path:
            return None
        if hasattr(repo_graph, 'root_dir') and repo_graph.root_dir:
            try:
                return os.path.relpath(file_path, repo_graph.root_dir).replace(os.sep, '/')
            except:
                pass
        return os.path.basename(file_path)

    if func_name in repo_graph.graph:
        return get_display_path(repo_graph.graph.nodes[func_name])
    for node in repo_graph.graph.nodes():
        if node.endswith(f".{func_name}") or node == func_name:
            path = get_display_path(repo_graph.graph.nodes[node])
            if path:
                return path
    return None

def node_to_simplified_code(node):
    if isinstance(node, ast.AST):
        try:
            # Handle control structures by only printing the header
            if isinstance(node, ast.If):
                test = ast.unparse(node.test)
                return f"if {test}:"
            elif isinstance(node, ast.While):
                test = ast.unparse(node.test)
                return f"while {test}:"
            elif isinstance(node, ast.For):
                target = ast.unparse(node.target)
                iter_ = ast.unparse(node.iter)
                return f"for {target} in {iter_}:"
            elif isinstance(node, ast.AsyncFor):
                target = ast.unparse(node.target)
                iter_ = ast.unparse(node.iter)
                return f"async for {target} in {iter_}:"
            elif isinstance(node, ast.FunctionDef):
                return f"def {node.name}(...):"
            elif isinstance(node, ast.AsyncFunctionDef):
                return f"async def {node.name}(...):"
            elif isinstance(node, ast.ClassDef):
                return f"class {node.name}:"
            elif isinstance(node, ast.With):
                 items = ", ".join([ast.unparse(i) for i in node.items])
                 return f"with {items}:"
            elif isinstance(node, ast.AsyncWith):
                 items = ", ".join([ast.unparse(i) for i in node.items])
                 return f"async with {items}:"
            elif isinstance(node, ast.Try):
                return "try:"

            return ast.unparse(node).strip()
        except:
            pass
    return str(node).strip()

def get_block_label(block):
    """
    Generates a descriptive label for a CFG block.
    """
    if not block.statements:
        return f"Block_{block.id}"

    stmt = block.statements[0]
    lineno = getattr(stmt, 'lineno', '?')

    # Try to derive a name from content
    try:
        code = node_to_simplified_code(stmt).split('\n')[0]
        # Keep it short
        if len(code) > 25:
            code = code[:22] + "..."
        # Sanitize for readability (allow alphanumeric and underscores)
        clean_code = "".join(c if c.isalnum() else "_" for c in code)
        # remove consecutive underscores
        while "__" in clean_code:
            clean_code = clean_code.replace("__", "_")
        return f"line{lineno}_{clean_code}"
    except:
        return f"line{lineno}_block{block.id}"

def get_cfg_prompt(code):
    """
    Generates a compliant Control Flow Graph (CFG) prompt.
    Rules:
    - Always start: ENTRY
    - Always end: EXIT
    - Flow operator: -> (space-arrow-space)
    - Condition check: check[condition]
    - Branch: -> {TRUE: true_path | FALSE: false_path}
    """
    try:
        cfg_builder = CFGBuilder()
        cfg = cfg_builder.build_from_src("input_code", code)

        lines = []

        for func_name, func_cfg in cfg.functioncfgs.items():
            lines.append(f"Function: {func_name}")
            lines.append("Always start: ENTRY")

            blocks = list(func_cfg.own_blocks())
            blocks.sort(key=lambda b: b.id)

            if not blocks:
                lines.append("Always end: EXIT\n")
                continue

            for block in blocks:
                stmts = [node_to_simplified_code(s) for s in block.statements]
                block_content = ", ".join(stmts)

                # Check for multiple exits (branching)
                if len(block.exits) > 1:
                    block_content += " check[condition]"

                line = f"Block {block.id}: {block_content}"

                if not block.exits:
                    line += " -> EXIT"
                elif len(block.exits) == 1:
                    target = block.exits[0].target.id
                    line += f" -> Block {target}"
                else:
                    # Branch handling
                    branches = []
                    for exit in block.exits:
                        label = "CASE"
                        # Handle exitcase
                        if hasattr(exit, 'exitcase') and exit.exitcase:
                            try:
                                label = ast.unparse(exit.exitcase)
                            except:
                                label = str(exit.exitcase)

                        branches.append(f"{label.upper()}: Block {exit.target.id}")

                    line += " -> {" + " | ".join(branches) + "}"

                lines.append(line)
            
            lines.append("Always end: EXIT\n")
            
        return "\n".join(lines)
    except Exception as e:
        return f"Error generating CFG prompt: {e}"

def get_pdg_prompt(code):
    """
    Generates a compliant Program Dependence Graph (PDG) prompt.
    Rules:
    - [DATA] and [CONTROL] sections
    - Data dependency: variable->location1,location2
    - Control dependency: statement<-condition
    - Descriptive location naming
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

            # 1. Compute Labels and display nodes
            block_labels = {b.id: get_block_label(b) for b in blocks}

            for block in blocks:
                stmts = [node_to_simplified_code(s) for s in block.statements]
                content = "; ".join(stmts)
                lines.append(f"Node {block_labels[block.id]}: {content}")

            # 2. Build Graphs
            entry_block_id = blocks[0].id
            nx_graph = nx.DiGraph()
            for block in blocks:
                nx_graph.add_node(block.id)
                for exit in block.exits:
                    nx_graph.add_edge(block.id, exit.target.id)
            
            # cd_edges: list of (controller, controlled)
            cd_edges = compute_control_dependencies(nx_graph, entry_block_id)
            dd_edges = compute_data_dependencies(blocks, nx_graph)
            
            # 3. [CONTROL] Section
            lines.append("\n[CONTROL]")
            # Output: controlled <- controller

            valid_ids = set(b.id for b in blocks)

            for controller, controlled in cd_edges:
                if controller in valid_ids and controlled in valid_ids:
                    ctrl_label = block_labels[controller]
                    stmt_label = block_labels[controlled]
                    # Format: Statement <- Condition
                    lines.append(f"{stmt_label} <- {ctrl_label}")

            # 4. [DATA] Section
            lines.append("\n[DATA]")
            # Group by (src, var) -> [dests]
            data_map = {}
            for src, dest, var in dd_edges:
                if src in valid_ids and dest in valid_ids:
                    key = (src, var)
                    if key not in data_map: data_map[key] = []
                    data_map[key].append(dest)
            
            for (src, var), dests in data_map.items():
                src_label = block_labels[src]
                dest_labels = ", ".join([block_labels[d] for d in dests])
                # Format: variable: def@location -> use@location1,location2
                lines.append(f"{var}: def@{src_label} -> use@{dest_labels}")
                
            lines.append("")

        return "\n".join(lines)
    except Exception as e:
        return f"Error generating PDG prompt: {e}"

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
    return get_cfg_prompt(code)

def get_pdg(code):
    return get_pdg_prompt(code)
