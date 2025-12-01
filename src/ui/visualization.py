import networkx as nx
import graphviz

def visualize_repo_graph(graph, max_nodes=50):
    """
    Visualizes the repository call graph using graphviz.
    Handles large graphs by limiting nodes or clustering.

    Args:
        graph (networkx.DiGraph): The full repo graph.
        max_nodes (int): Maximum nodes to render to prevent UI freeze.

    Returns:
        graphviz.Digraph: The graphviz object or None if error.
    """
    try:
        dot = graphviz.Digraph(comment='Repository Call Graph')
        dot.attr(rankdir='LR') # Left to Right layout
        dot.attr('node', shape='box', style='filled', fillcolor='#E0F7FA', fontname='Helvetica')
        dot.attr('edge', color='#546E7A')

        # If graph is too large, visualize the most important nodes (highest degree)
        if len(graph.nodes) > max_nodes:
            # Calculate degree (in + out)
            degrees = dict(graph.degree())
            # Sort by degree descending
            top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
            subgraph = graph.subgraph(top_nodes)

            # Add a visual note that this is a subset
            dot.attr(label=f"Top {max_nodes} Nodes by Connectivity (Full Graph: {len(graph.nodes)} nodes)",labelloc='t')
        else:
            subgraph = graph

        # Add Nodes
        for node in subgraph.nodes():
            # Label with simple name if possible to save space, or full name
            label = str(node)
            dot.node(label, label)

        # Add Edges
        for u, v in subgraph.edges():
            dot.edge(str(u), str(v))

        return dot
    except Exception as e:
        print(f"Error visualizing repo graph: {e}")
        return None

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
    return visualize_repo_graph(graph) # Reuse the better visualizer
