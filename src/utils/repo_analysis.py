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

    def build_call_graph(self):
        """
        Builds a call graph for all Python files in the repository.
        """
        graph = nx.DiGraph()
        python_files = self.find_python_files()

        # To store all function and class method definitions
        definitions = {}
        # To store all imports in each file
        imports = {}

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

        # Second pass: find all imports
        for file_path in python_files:
            full_path = os.path.join(self.target_dir, file_path)
            file_imports = {}
            with open(full_path, "r", encoding="utf-8") as f:
                source = f.read()
                try:
                    tree = ast.parse(source)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                file_imports[alias.asname or alias.name] = alias.name
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                for alias in node.names:
                                    file_imports[alias.asname or alias.name] = f"{node.module}.{alias.name}"
                    imports[file_path] = file_imports
                except Exception as e:
                    print(f"Error parsing {file_path} for imports: {e}")

        # Third pass: find all calls
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


if __name__ == "__main__":
    # Test with a small repo
    analyzer = RepoAnalyzer("https://github.com/psf/requests")
    # analyzer.clone_repo()
    # call_graph = analyzer.build_call_graph()
    # print(call_graph.nodes)
    # print(call_graph.edges)
