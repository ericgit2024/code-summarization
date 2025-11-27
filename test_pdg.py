from src.structure.graph_utils import get_pdg

code = """
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
"""

print("Generating PDG for factorial function...")
pdg = get_pdg(code)
print(pdg)
