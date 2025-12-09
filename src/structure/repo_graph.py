import os
import ast
import networkx as nx
import logging
from src.structure.ast_analyzer import ASTAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RepoGraphBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_metadata = {} # Cache for metadata

    def add_function(self, name, code, file_path, docstring=None, metadata=None):
        """Adds a function node to the graph with enhanced metadata."""
        # Use simple name for ID, but store rich metadata
        # If metadata has qualified name (e.g. Class.method), we might want to use that as ID.
        # But for backward compatibility/simplicity, we might use just function name or fully qualified if possible.
        # Let's try to use the name provided which should be fully qualified if coming from ASTAnalyzer

        node_id = name
        self.graph.add_node(
            node_id,
            code=code,
            file_path=file_path,
            docstring=docstring,
            type="function",
            metadata=metadata or {}
        )
        logger.debug(f"Added function node: {node_id}")

    def build_from_directory(self, root_dir):
        """Walks the directory and parses all Python files using ASTAnalyzer."""
        logger.info(f"Building graph from directory: {root_dir}")
        count = 0
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            code = f.read()
                        self._parse_and_add(code, file_path)
                        count += 1
                    except Exception as e:
                        logger.error(f"Failed to read {file_path}: {e}")
        logger.info(f"Parsed {count} files from directory.")
        self._build_edges()

    def build_from_file(self, file_path):
        """Parses a single file using ASTAnalyzer."""
        logger.info(f"Building graph from file: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            self._parse_and_add(content, file_path)
            self._build_edges()
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")

    def _parse_and_add(self, code, file_path):
        """Parses code using ASTAnalyzer and adds functions to graph."""
        try:
            analyzer = ASTAnalyzer(code)
            analysis_result = analyzer.analyze()

            # Store file-level imports in the graph (maybe as a special node or just map)
            # For now, we attach imports to every function node in that file to help resolution
            file_imports = analysis_result["imports"]

            # Add functions
            for func_name, meta in analysis_result["functions"].items():
                meta["file_imports"] = file_imports # Store imports for resolution
                self.add_function(
                    func_name,
                    meta.get("source", ""),
                    file_path,
                    meta.get("docstring"),
                    metadata=meta
                )

            logger.info(f"Extracted {len(analysis_result['functions'])} functions from {file_path}")
        except Exception as e:
            logger.error(f"Error parsing code from {file_path}: {e}")

    def _build_edges(self):
        """
        Iterates over all nodes and adds call edges based on AST analysis.
        Handles imports and class methods resolution.
        """
        logger.info("Building graph edges with enhanced resolution...")
        nodes_data = list(self.graph.nodes(data=True))
        edge_count = 0

        # Create a lookup map for function names
        # Map simple names to list of potential full names (to handle ambiguity or lack of imports)
        name_lookup = {}
        for node in self.graph.nodes():
            simple_name = node.split('.')[-1]
            if simple_name not in name_lookup:
                name_lookup[simple_name] = []
            name_lookup[simple_name].append(node)

        for node, data in nodes_data:
            if data.get("type") != "function": continue

            meta = data.get("metadata", {})
            calls = meta.get("calls", []) # List of {name, context}
            imports = meta.get("file_imports", [])

            for call in calls:
                call_name = call["name"]
                context = call.get("context", [])

                target_node = None
                
                # 1. Direct match (e.g. internal function call or fully qualified)
                if call_name in self.graph:
                    target_node = call_name
                
                # 2. Local method call (self.method)
                elif call_name.startswith("self."):
                    method_name = call_name.split(".")[1]
                    # Assuming node is Class.method, we want Class.method_name
                    if "." in node:
                        class_prefix = node.rsplit(".", 1)[0]
                        potential_target = f"{class_prefix}.{method_name}"
                        if potential_target in self.graph:
                            target_node = potential_target

                # 3. Simple name match in same file (ignoring classes for a moment)
                # If we call 'foo', and 'foo' is defined in the same file/scope
                # But 'node' might be qualified.

                # 4. Import resolution
                # If call_name is 'other_mod.func', check imports
                if not target_node:
                    target_node = self._resolve_import(call_name, imports)

                # 5. Fallback: Naive name matching if unique
                if not target_node:
                    simple_call_name = call_name.split('.')[-1]
                    candidates = name_lookup.get(simple_call_name, [])
                    if len(candidates) == 1:
                        target_node = candidates[0]
                    elif len(candidates) > 1:
                        # Ambiguous, maybe pick the one in same file?
                        same_file_candidates = [c for c in candidates if self.graph.nodes[c]['file_path'] == data['file_path']]
                        if len(same_file_candidates) == 1:
                            target_node = same_file_candidates[0]

                if target_node and target_node != node:
                    # Add edge with context
                    self.graph.add_edge(node, target_node, type="calls", context=context)
                    edge_count += 1

        logger.info(f"Built {edge_count} edges in the graph.")

    def _resolve_import(self, call_name, imports):
        """
        Resolves a function call name using the file's imports.
        """
        parts = call_name.split('.')
        base = parts[0]

        for imp in imports:
            # from module import name [as alias]
            if "name" in imp:
                alias = imp.get("alias") or imp["name"]
                if alias == base:
                    # Resolved to module.name
                    # If call was 'alias.func', it maps to 'module.name.func' ??
                    # If call was 'alias' (and it's a function), it maps to 'module.name'

                    # Case 1: from x import func as f; f() -> x.func
                    if len(parts) == 1:
                         full_name = f"{imp['module']}.{imp['name']}"
                         # We need to check if 'full_name' is a node in our graph
                         # Our graph nodes are usually just 'func' or 'Class.method'
                         # because we parse file by file.
                         # We don't store module prefix in node names currently unless we change add_function.
                         # Wait, add_function uses 'name' from AST.
                         # If parsing file 'src/utils/foo.py', function 'bar' is named 'bar'.
                         # We don't verify module path. This is a limitation.
                         # But let's check if we can match against candidates.
                         return self._find_node_by_module_match(imp['module'], imp['name'])

            # import module [as alias]
            elif "module" in imp and "name" not in imp:
                alias = imp.get("alias") or imp["module"]
                if alias == base:
                     # import utils; utils.helper() -> helper in utils.py
                     if len(parts) > 1:
                         # module is imp['module'], func is parts[1:]
                         func_part = ".".join(parts[1:])
                         return self._find_node_by_module_match(imp['module'], func_part)
        return None

    def _find_node_by_module_match(self, module_name, func_name):
        """
        Tries to find a node that matches the module and function name.
        Since we don't store module paths explicitly in node IDs, we check file paths.
        module_name: e.g. 'src.utils.helper' or 'utils'
        """
        # Convert dotted module to path part
        path_part = module_name.replace('.', os.sep)

        for node, data in self.graph.nodes(data=True):
            if node.endswith(func_name) or node == func_name:
                file_path = data.get('file_path', '')
                # Check if file path ends with module path
                # e.g. path_part='utils', file_path='.../src/utils.py'
                # or path_part='src/utils', file_path='.../src/utils.py'
                if path_part in file_path:
                    # Check if the function name matches
                    # node could be 'Class.method', func_name 'Class.method'
                    if node == func_name:
                        return node
                    # If func_name is simple 'func', and node is 'func'
        return None

    def extract_dependency_subgraph(self, target_node, max_nodes=10):
        """
        Intelligently extracts a subgraph of relevant dependencies.
        Uses scoring based on proximity, complexity, shared variables, and control flow.
        """
        if target_node not in self.graph:
            return None

        scored_nodes = []
        visited = set()
        queue = [(target_node, 0)] # (node, distance)
        visited.add(target_node)

        # 1. Collect candidates (BFS up to depth 2 or 3 to get candidates)
        candidates = set()
        for _ in range(2): # 2 hops
             next_layer = []
             for n, dist in queue:
                 # Outgoing edges (dependencies)
                 for neighbor in self.graph.successors(n):
                     if neighbor not in visited:
                         visited.add(neighbor)
                         candidates.add(neighbor)
                         next_layer.append((neighbor, dist + 1))
             queue = next_layer

        # 2. Score candidates
        for node in candidates:
            score = self._calculate_relevance_score(target_node, node)
            scored_nodes.append((node, score))

        # 3. Sort and Select
        scored_nodes.sort(key=lambda x: x[1]['total'], reverse=True)
        selected_nodes = [target_node] + [n for n, s in scored_nodes[:max_nodes]]
        
        subgraph = self.graph.subgraph(selected_nodes)

        # Attach scores to the subgraph nodes for visualization/prompting
        for n, s in scored_nodes:
            if n in subgraph:
                subgraph.nodes[n]['relevance_score'] = s
        
        return subgraph

    def _calculate_relevance_score(self, source, target):
        """
        Calculates relevance of target to source.
        Returns dict with total score and breakdown.
        """
        score = 0
        breakdown = []

        # 1. Proximity
        try:
            distance = nx.shortest_path_length(self.graph, source, target)
            prox_score = 1.0 / distance if distance > 0 else 1.0
            score += prox_score * 3.0 # Weight 3
            breakdown.append(f"Proximity: {prox_score:.2f}")
        except:
            pass # No path?

        target_data = self.graph.nodes[target]
        target_meta = target_data.get("metadata", {})

        # 2. Complexity
        comp = target_meta.get("complexity", {})
        cyclomatic = comp.get("cyclomatic", 1)
        # We prefer somewhat complex functions (logic) over trivial getters
        # But too complex might be distracting?
        # Prompt says: "cyclomatic complexity (avoid including trivial helper functions)"
        # So higher complexity is better.
        comp_score = min(cyclomatic, 10) / 10.0 # Normalize 1-10
        score += comp_score * 2.0
        breakdown.append(f"Complexity: {comp_score:.2f}")

        # 3. Control Flow Importance
        # Check edge attributes
        edge_data = self.graph.get_edge_data(source, target)
        if edge_data:
            context = edge_data.get("context", [])
            if "loop" in context:
                score += 2.0
                breakdown.append("In Loop")
            if "branch" in context:
                score += 1.0
                breakdown.append("In Branch")

        # 4. Data Flow (Shared Variables?)
        # This requires looking at variable usage in both functions.
        # We stored 'variables' -> 'used'/'defined' in metadata.
        source_vars = set(self.graph.nodes[source].get("metadata", {}).get("variables", {}).get("used", []))
        target_vars = set(target_meta.get("variables", {}).get("used", []))
        # If they use same variables (global or similar names), might be related.
        shared = source_vars.intersection(target_vars)
        if shared:
            score += len(shared) * 0.5
            breakdown.append(f"Shared Vars: {len(shared)}")

        return {"total": score, "breakdown": ", ".join(breakdown)}

    def get_context_text(self, node_name, depth=None): # Depth ignored if using smart extraction
        """
        Generates a text description using intelligent subgraph extraction.
        """
        subgraph = self.extract_dependency_subgraph(node_name)
        if not subgraph:
            return "No context found."

        lines = []
        lines.append(f"Context for function '{node_name}':")
        lines.append("")
        lines.append("**IMPORTANT**: When mentioning these functions in your summary, ALWAYS include their source file using the format: 'function_name() from filename.py'")
        lines.append("")
        
        # List selected dependencies with scores
        for n in subgraph.nodes():
            if n == node_name: continue
            data = subgraph.nodes[n]
            score_info = data.get("relevance_score", {})
            total = score_info.get("total", 0)
            breakdown = score_info.get("breakdown", "N/A")

            doc = data.get("docstring")
            doc_summary = doc.split('\n')[0] if doc else "No docstring"
            
            file_path = data.get("file_path", "unknown file")
            filename = os.path.basename(file_path)

            lines.append(f"  - **{n}()** from **{filename}** (Relevance: {total:.1f})")
            lines.append(f"    Description: {doc_summary}")
            lines.append(f"    Reason: {breakdown}")

            # Maybe show signature?
            meta = data.get("metadata", {})
            args = ", ".join([a['name'] for a in meta.get("args", [])])
            lines.append(f"    Signature: def {n.split('.')[-1]}({args})")
            lines.append("")

        return "\n".join(lines)
