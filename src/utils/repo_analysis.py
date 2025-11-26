import os
import git
import ast

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

if __name__ == "__main__":
    # Test with a small repo
    analyzer = RepoAnalyzer("https://github.com/psf/requests")
    # analyzer.clone_repo() # Commented out to avoid cloning in CI environment unless necessary
    # print("Dependencies extracted.")
