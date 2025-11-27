from py2cfg import CFGBuilder
import inspect

code = """
def foo(x):
    return x
"""

cfg_builder = CFGBuilder()
cfg = cfg_builder.build_from_src("test", code)

for func_name, func_cfg in cfg.functioncfgs.items():
    print(f"Function: {func_name}")
    print(f"Attributes: {dir(func_cfg)}")
    # Try to find entry block
    if hasattr(func_cfg, 'entry_block'):
        print(f"Entry block: {func_cfg.entry_block}")
    else:
        print("No entry_block attribute found.")
