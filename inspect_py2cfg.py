from py2cfg import CFGBuilder
import inspect

code = """
x = 1
def test(x):
    return x
"""

try:
    cfg_builder = CFGBuilder()
    cfg = cfg_builder.build_from_src("test", code)
    
    print("Top level blocks:")
    for block in cfg.own_blocks():
        print(block.id)
        
    print("Function blocks:")
    for func_name, func_cfg in cfg.functioncfgs.items():
        print(f"Function: {func_name}")
        for block in func_cfg.own_blocks():
            print(block.id)
            
except Exception as e:
    print(e)
