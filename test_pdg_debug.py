from src.structure.graph_utils import get_pdg
from py2cfg import CFGBuilder

code = """
class Math:
    def factorial(self, n):
        if n == 0:
            return 1
        else:
            return n * self.factorial(n-1)

def standalone(x):
    return x + 1
"""

cfg_builder = CFGBuilder()
cfg = cfg_builder.build_from_src("input_code", code)
print("Keys in functioncfgs:", list(cfg.functioncfgs.keys()))
