import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.structure.repo_graph import RepoGraphBuilder
from src.data.prompt import construct_prompt

def test_graph_pipeline():
    print("Testing RepoGraphBuilder...")
    
    dummy_code = """
def helper_a():
    return "A"

def helper_b():
    return helper_a()

def main():
    res = helper_b()
    print(res)
"""
    filename = "test_repo_dump.py"
    with open(filename, "w") as f:
        f.write(dummy_code)
        
    builder = RepoGraphBuilder()
    builder.build_from_file(filename)
    
    # Check nodes
    print(f"Nodes: {builder.graph.nodes()}")
    assert "main" in builder.graph
    assert "helper_b" in builder.graph
    assert "helper_a" in builder.graph
    
    # Check edges
    print(f"Edges: {builder.graph.edges()}")
    assert builder.graph.has_edge("main", "helper_b")
    assert builder.graph.has_edge("helper_b", "helper_a")
    
    # Check context text
    context = builder.get_context_text("helper_b")
    print(f"\nContext for 'helper_b':\n{context}")
    assert "Called by: main" in context
    assert "Calls: helper_a" in context
    
    # Check prompt construction
    print("\nTesting construct_prompt...")
    prompt = construct_prompt(
        structural_prompt="[AST]...",
        query_code="def helper_b():...",
        retrieved_codes=[],
        retrieved_docstrings=[],
        repo_context=context
    )
    
    print(f"Generated Prompt Snippet:\n{prompt[:200]}...")
    assert "Repository Context:" in prompt
    assert "Called by: main" in prompt
    
    print("\nSUCCESS: Graph pipeline verified!")
    
    if os.path.exists(filename):
        os.remove(filename)

if __name__ == "__main__":
    test_graph_pipeline()
