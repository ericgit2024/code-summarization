from src.structure.graph_utils import get_call_graph

code = """
def foo():
    bar()
    baz(1, 2)

def main():
    foo()
    print("Done")
"""

print("--- Code ---")
print(code)
print("\n--- Call Graph ---")
print(get_call_graph(code))
