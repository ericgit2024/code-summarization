import os
import git
import ast
import networkx as nx

class RepoAnalyzer:
    def __init__(self, repo_url, target_dir="downloaded_repo"):
        self.repo_url = repo_url
        self.target_dir = target_dir

    def clone_repo(self):
        if os.path.exists(self.target_dir):
            print(f"Directory {self.target_dir} exists. Using existing repo.")
        else:
            print(f"Cloning {self.repo_url}...")
            git.Repo.clone_from(self.repo_url, self.target_dir)

    def get_dependencies(self):
        """
        Analyzes imports in the repo to build a simple dependency graph.
        Returns a dictionary mapping files to their imports.
        """
        dependencies = {}
        for root, dirs, files in os.walk(self.target_dir):
            for file in files:
                if file.endswith(".py"):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, "r") as f:
                            tree = ast.parse(f.read())

                        imports = []
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Import):
                                for alias in node.names:
                                    imports.append(alias.name)
                            elif isinstance(node, ast.ImportFrom):
                                module = node.module if node.module else ""
                                imports.append(module)

                        rel_path = os.path.relpath(filepath, self.target_dir)
                        dependencies[rel_path] = imports
                    except Exception as e:
                        print(f"Error parsing {filepath}: {e}")
        return dependencies

    def find_python_files(self):
        """
        Scans the repository for all .py files.
        """
        python_files = []
        for root, _, files in os.walk(self.target_dir):
            for file in files:
                if file.endswith(".py"):
                    python_files.append(os.path.relpath(os.path.join(root, file), self.target_dir))
        return python_files

    def extract_function_code(self, file_path, function_name):
        """
        Extracts the source code of a specific function from a file.
        """
        try:
            full_path = os.path.join(self.target_dir, file_path)
            with open(full_path, "r", encoding="utf-8") as f:
                source = f.read()
                tree = ast.parse(source)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    return ast.get_source_segment(source, node)
            return None # Function not found
        except Exception as e:
            print(f"Error extracting function {function_name} from {file_path}: {e}")
            return None

    def extract_class_code(self, file_path, class_name):
        """
        Extracts the source code of a specific class from a file.
        """
        try:
            full_path = os.path.join(self.target_dir, file_path)
            with open(full_path, "r", encoding="utf-8") as f:
                source = f.read()
                tree = ast.parse(source)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    return ast.get_source_segment(source, node)
            return None # Class not found
        except Exception as e:
            print(f"Error extracting class {class_name} from {file_path}: {e}")
            return None

    def extract_file_code(self, file_path):
        """
        Extracts the source code of a specific file.
        """
        try:
            full_path = os.path.join(self.target_dir, file_path)
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None

    def build_call_graph(self):
        """
        Builds a call graph for all Python files in the repository.
        TODO: This is a simplified implementation and does not resolve calls across files.
        """
        graph = nx.DiGraph()
        python_files = self.find_python_files()

        # To store all function and class method definitions
        definitions = {}

        # First pass: find all function, class, and method definitions
        for file_path in python_files:
            full_path = os.path.join(self.target_dir, file_path)
            with open(full_path, "r", encoding="utf-8") as f:
                source = f.read()
                try:
                    tree = ast.parse(source)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            definitions[node.name] = file_path
                            graph.add_node(node.name)
                        elif isinstance(node, ast.ClassDef):
                            for method in node.body:
                                if isinstance(method, ast.FunctionDef):
                                    method_name = f"{node.name}.{method.name}"
                                    definitions[method_name] = file_path
                                    graph.add_node(method_name)
                except Exception as e:
                    print(f"Error parsing {file_path} for definitions: {e}")

        # Second pass: find all calls
        for file_path in python_files:
            full_path = os.path.join(self.target_dir, file_path)
            with open(full_path, "r", encoding="utf-8") as f:
                source = f.read()
                try:
                    tree = ast.parse(source)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                            parent_name = node.name if isinstance(node, ast.ClassDef) else ""
                            for sub_node in node.body:
                                if isinstance(sub_node, ast.FunctionDef):
                                    caller_name = f"{parent_name}.{sub_node.name}" if parent_name else sub_node.name
                                    for call_node in ast.walk(sub_node):
                                        if isinstance(call_node, ast.Call):
                                            callee_name = self._resolve_call_name(call_node.func)
                                            if callee_name and callee_name in definitions:
                                                graph.add_edge(caller_name, callee_name)
                except Exception as e:
                    print(f"Error parsing {file_path} for calls: {e}")
        return graph

    def _resolve_call_name(self, call_node):
        if isinstance(call_node, ast.Name):
            return call_node.id
        elif isinstance(call_node, ast.Attribute):
            # Recursively resolve the attribute chain
            value = self._resolve_call_name(call_node.value)
            if value:
                return f"{value}.{call_node.attr}"
        return None

    def analyze_graph_metrics(self, graph):
        """
        Calculates and returns key metrics for a given graph.
        """
        if not isinstance(graph, nx.DiGraph):
            return "Not a valid graph object."

        metrics = {
            "Number of Nodes": graph.number_of_nodes(),
            "Number of Edges": graph.number_of_edges(),
            "Density": nx.density(graph) if graph.number_of_nodes() > 0 else 0,
        }

        if graph.number_of_nodes() > 0:
            # Degree Centrality
            degree_centrality = nx.degree_centrality(graph)
            # Get top 5 nodes by degree centrality
            top_5_centrality = sorted(degree_centrality.items(), key=lambda item: item[1], reverse=True)[:5]
            metrics["Top 5 Nodes by Degree Centrality"] = top_5_centrality

        return metrics


if __name__ == "__main__":
    # Test with a small repo
    analyzer = RepoAnalyzer("https://github.com/psf/requests")
    # analyzer.clone_repo()
    # call_graph = analyzer.build_call_graph()
    # print(call_graph.nodes)
    # print(call_graph.edges)
