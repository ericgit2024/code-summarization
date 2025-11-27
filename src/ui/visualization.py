import networkx as nx
import graphviz

def visualize_dependency_graph(dependencies):
    """
    Visualizes the dependency graph using graphviz.

    Args:
        dependencies (dict): A dictionary mapping files to their imports.

    Returns:
        graphviz.Digraph: The graphviz object or None if error.
    """
    try:
        dot = graphviz.Digraph(comment='Dependency Graph')
        dot.attr(rankdir='LR')

        for file, imports in dependencies.items():
            dot.node(file, file)
            for imp in imports:
                dot.edge(file, imp)

        return dot
    except Exception as e:
        print(f"Error visualizing dependency graph: {e}")
        return None

def visualize_call_graph(graph):
    """
    Visualizes the call graph using graphviz.

    Args:
        graph (networkx.DiGraph): The call graph.

    Returns:
        graphviz.Digraph: The graphviz object or None if error.
    """
    try:
        dot = graphviz.Digraph(comment='Call Graph')
        dot.attr(rankdir='LR')

        for node in graph.nodes():
            dot.node(str(node), str(node))

        for u, v in graph.edges():
            dot.edge(str(u), str(v))

        return dot
    except Exception as e:
        print(f"Error visualizing call graph: {e}")
        return None
