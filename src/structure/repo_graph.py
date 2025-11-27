import os
import ast
import networkx as nx
import logging
from src.structure.graph_utils import extract_call_graph_edges

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RepoGraphBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_function(self, name, code, file_path, docstring=None):
        """Adds a function node to the graph."""
        # Use a unique identifier if possible, but for now simple name
        self.graph.add_node(name, code=code, file_path=file_path, docstring=docstring, type="function")
        logger.debug(f"Added function node: {name}")

    def build_from_directory(self, root_dir):
        """Walks the directory and parses all Python files."""
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
        """Parses a single file."""
        logger.info(f"Building graph from file: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            self._parse_and_add(content, file_path)
            self._build_edges()
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")

    def _parse_and_add(self, code, file_path):
        """Parses code to find function definitions and adds them to the graph."""
        try:
            tree = ast.parse(code)
            func_count = 0
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_name = node.name
                    # Try to get source code
                    try:
                        func_code = ast.get_source_segment(code, node)
                    except:
                        func_code = f"def {func_name}(...): pass # Source not available"
                    
                    docstring = ast.get_docstring(node)
                    self.add_function(func_name, func_code, file_path, docstring)
                    func_count += 1
            logger.info(f"Extracted {func_count} functions from {file_path}")
        except Exception as e:
            logger.error(f"Error parsing code from {file_path}: {e}")

    def _build_edges(self):
        """Iterates over all nodes and adds call edges."""
        logger.info("Building graph edges...")
        nodes_data = list(self.graph.nodes(data=True))
        edge_count = 0
        for node, data in nodes_data:
            if data.get("type") == "function":
                code = data.get("code")
                if not code: continue
                
                calls = extract_call_graph_edges(code)
                # calls is {caller_name: [callee_names]}
                
                for caller, callees in calls.items():
                    # In a single function snippet, caller is the function itself
                    # We assume 'node' matches 'caller' (function name)
                    for callee in callees:
                        if callee in self.graph and callee != node:
                            self.graph.add_edge(node, callee, type="calls")
                            edge_count += 1
                            logger.debug(f"Added edge: {node} -> {callee}")
        logger.info(f"Built {edge_count} edges in the graph.")

    def get_subgraph(self, node_name, depth=1):
        """Retrieves a subgraph centered around the node."""
        logger.info(f"Retrieving subgraph for '{node_name}' with depth {depth}")
        if node_name not in self.graph:
            logger.warning(f"Node '{node_name}' not found in graph.")
            return None
        
        nodes = {node_name}
        current_layer = {node_name}
        
        # BFS for depth
        for d in range(depth):
            next_layer = set()
            for n in current_layer:
                # Add successors (callees)
                succs = list(self.graph.successors(n))
                next_layer.update(succs)
                # Add predecessors (callers)
                preds = list(self.graph.predecessors(n))
                next_layer.update(preds)
            
            nodes.update(next_layer)
            current_layer = next_layer
            logger.debug(f"Depth {d+1}: Found {len(next_layer)} neighbors.")
            
        subgraph = self.graph.subgraph(nodes)
        logger.info(f"Subgraph constructed with {len(subgraph.nodes)} nodes and {len(subgraph.edges)} edges.")
        return subgraph

    def get_context_text(self, node_name, depth=1):
        """Generates a text description of the subgraph context."""
        subgraph = self.get_subgraph(node_name, depth)
        if not subgraph:
            return "No context found."
            
        lines = []
        lines.append(f"Context for function '{node_name}':")
        
        # List Callers
        callers = [p for p in subgraph.predecessors(node_name)]
        if callers:
            lines.append(f"  Called by: {', '.join(callers)}")
            logger.info(f"Found {len(callers)} callers for {node_name}")
            
        # List Callees
        callees = [s for s in subgraph.successors(node_name)]
        if callees:
            lines.append(f"  Calls: {', '.join(callees)}")
            logger.info(f"Found {len(callees)} callees for {node_name}")
            
        # Add details for neighbors
        for n in subgraph.nodes():
            if n == node_name: continue
            data = subgraph.nodes[n]
            doc = data.get("docstring")
            doc_summary = doc.split('\n')[0] if doc else "No docstring"
            lines.append(f"  - Function '{n}': {doc_summary}")
            
        return "\n".join(lines)
