from src.structure.graph_utils import get_pdg

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

print("Generating PDG for class method and standalone function...")
print(get_pdg(code))
