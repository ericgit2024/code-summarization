import ast
import logging

logger = logging.getLogger(__name__)

class ASTAnalyzer(ast.NodeVisitor):
    def __init__(self, source_code):
        self.source_code = source_code
        self.tree = ast.parse(source_code)
        self.functions = {}
        self.classes = {}
        self.imports = []
        self.current_scope = None

    def analyze(self):
        """
        Analyzes the source code and returns comprehensive metadata.
        """
        self.visit(self.tree)
        return {
            "functions": self.functions,
            "classes": self.classes,
            "imports": self.imports
        }

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append({"module": alias.name, "alias": alias.asname})
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        module = node.module
        for alias in node.names:
            self.imports.append({"module": module, "name": alias.name, "alias": alias.asname})
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        class_info = {
            "name": node.name,
            "lineno": node.lineno,
            "end_lineno": getattr(node, 'end_lineno', -1),
            "docstring": ast.get_docstring(node),
            "bases": [self._get_name(b) for b in node.bases],
            "methods": []
        }

        # Analyze methods within the class
        old_scope = self.current_scope
        self.current_scope = node.name

        for item in node.body:
            if isinstance(item, ast.FunctionDef) or isinstance(item, ast.AsyncFunctionDef):
                method_info = self._analyze_function(item, is_method=True)
                class_info["methods"].append(method_info)
                # Also add to global functions map with class prefix for easier lookup
                qualified_name = f"{node.name}.{item.name}"
                self.functions[qualified_name] = method_info

        self.current_scope = old_scope
        self.classes[node.name] = class_info
        # We don't call generic_visit to avoid double counting functions if we handled them here
        # But we might miss nested classes. For now, assume flat class structure mostly.

    def visit_FunctionDef(self, node):
        # Top level functions (or nested if we are visiting recursively)
        # If we are inside a class, it's handled by visit_ClassDef
        if self.current_scope is None:
            func_info = self._analyze_function(node, is_method=False)
            self.functions[node.name] = func_info
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        if self.current_scope is None:
            func_info = self._analyze_function(node, is_method=False)
            self.functions[node.name] = func_info
        self.generic_visit(node)

    def _analyze_function(self, node, is_method=False):
        """
        Deep analysis of a function node.
        """
        # 1. basic info
        info = {
            "name": node.name,
            "is_method": is_method,
            "lineno": node.lineno,
            "end_lineno": getattr(node, 'end_lineno', -1),
            "docstring": ast.get_docstring(node),
            "args": [],
            "returns": self._get_annotation(node.returns),
            "complexity": {},
            "calls": [],
            "variables": {"defined": set(), "used": set()},
            "control_structure": {"loops": 0, "branches": 0, "exceptions": 0}
        }

        # 2. Parameters
        for arg in node.args.args:
            info["args"].append({
                "name": arg.arg,
                "type": self._get_annotation(arg.annotation)
            })

        # 3. Analyze body for complexity, calls, vars
        body_analyzer = FunctionBodyAnalyzer()
        body_analyzer.visit(node)

        info["complexity"] = {
            "cyclomatic": body_analyzer.complexity,
            "loc": (info["end_lineno"] - info["lineno"] + 1) if info["end_lineno"] != -1 else len(node.body),
            "param_count": len(info["args"])
        }
        info["calls"] = body_analyzer.calls
        info["variables"]["defined"] = list(body_analyzer.defined_vars)
        info["variables"]["used"] = list(body_analyzer.used_vars)
        info["control_structure"] = body_analyzer.structure_counts

        # Source code if available
        try:
            info["source"] = ast.get_source_segment(self.source_code, node)
        except:
            info["source"] = None

        return info

    def _get_name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return "unknown"

    def _get_annotation(self, node):
        if node is None:
            return None
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Constant):
            return str(node.value)
        # Handle Subscript (e.g. List[int])
        elif isinstance(node, ast.Subscript):
            value = self._get_annotation(node.value)
            slice_val = self._get_annotation(node.slice)
            return f"{value}[{slice_val}]"
        return "complex_type"

class FunctionBodyAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.complexity = 1 # Base complexity
        self.calls = []
        self.defined_vars = set()
        self.used_vars = set()
        self.structure_counts = {"loops": 0, "branches": 0, "exceptions": 0}
        self.in_loop = 0
        self.in_branch = 0

    def visit_Call(self, node):
        func_name = self._get_func_name(node.func)
        context_flags = []
        if self.in_loop > 0: context_flags.append("loop")
        if self.in_branch > 0: context_flags.append("branch")

        self.calls.append({
            "name": func_name,
            "context": context_flags,
            "lineno": node.lineno
        })
        self.generic_visit(node)

    def _get_func_name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            # Handle self.method
            return f"{self._get_func_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Call):
             # Chained calls like f()()
            return "dynamic_call"
        return "unknown"

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            self.defined_vars.add(node.id)
        elif isinstance(node.ctx, ast.Load):
            self.used_vars.add(node.id)

    # Complexity & Structure Tracking
    def visit_If(self, node):
        self.complexity += 1
        self.structure_counts["branches"] += 1
        self.in_branch += 1
        self.generic_visit(node)
        self.in_branch -= 1

    def visit_For(self, node):
        self.complexity += 1
        self.structure_counts["loops"] += 1
        self.in_loop += 1
        self.generic_visit(node)
        self.in_loop -= 1

    def visit_AsyncFor(self, node):
        self.complexity += 1
        self.structure_counts["loops"] += 1
        self.in_loop += 1
        self.generic_visit(node)
        self.in_loop -= 1

    def visit_While(self, node):
        self.complexity += 1
        self.structure_counts["loops"] += 1
        self.in_loop += 1
        self.generic_visit(node)
        self.in_loop -= 1

    def visit_Try(self, node):
        self.structure_counts["exceptions"] += 1
        # Each except block is a decision point
        self.complexity += len(node.handlers)
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        # AND/OR operators increase complexity
        self.complexity += (len(node.values) - 1)
        self.generic_visit(node)

    # Helper for attribute names (e.g. self.x)
    def visit_Attribute(self, node):
        # We generally treat attributes as uses, unless in Store context
        # but simplified logic here:
        if isinstance(node.ctx, ast.Store):
             # For attributes, we might want to track 'self.x'
             name = self._get_func_name(node)
             self.defined_vars.add(name)
        elif isinstance(node.ctx, ast.Load):
             name = self._get_func_name(node)
             self.used_vars.add(name)
