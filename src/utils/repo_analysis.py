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

if __name__ == "__main__":
    # Test with a small repo
    analyzer = RepoAnalyzer("https://github.com/psf/requests")
    # analyzer.clone_repo() # Commented out to avoid cloning in CI environment unless necessary
    # print("Dependencies extracted.")
