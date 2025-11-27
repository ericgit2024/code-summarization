from py2cfg import CFGBuilder
import ast

code = """
def foo(x):
    y = x + 1
    return y
"""

cfg_builder = CFGBuilder()
cfg = cfg_builder.build_from_src("test", code)

for func_cfg in cfg.functioncfgs.values():
    for block in func_cfg.own_blocks():
        print(f"Block {block.id}:")
        for stmt in block.statements:
            print(f"  Type: {type(stmt)}")
            print(f"  Content: {stmt}")
            if isinstance(stmt, ast.AST):
                print("  Is AST node")
