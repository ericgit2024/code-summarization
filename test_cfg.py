from src.structure.graph_utils import visualize_cfg

code = """
def test(x):
    if x > 0:
        return x
    else:
        return -x
"""

dot = visualize_cfg(code)
if dot:
    print("CFG generated successfully.")
    print(dot.source)
else:
    print("Failed to generate CFG.")
