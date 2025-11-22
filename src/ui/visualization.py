import networkx as nx
import graphviz

def visualize_dependency_graph(dependencies, output_path="dependency_graph"):
    """
    Visualizes the dependency graph using graphviz.

    Args:
        dependencies (dict): A dictionary mapping files to their imports.
        output_path (str): Path to save the graph image.
    """
    G = nx.DiGraph()

    for file, imports in dependencies.items():
        G.add_node(file)
        for imp in imports:
            G.add_edge(file, imp)

    try:
        # Convert to Graphviz AGraph
        A = nx.nx_agraph.to_agraph(G)
        A.layout('dot')
        A.draw(f"{output_path}.png")
        print(f"Graph saved to {output_path}.png")
        return f"{output_path}.png"
    except ImportError:
        print("pygraphviz not installed. Please install it for visualization.")
        return None
    except Exception as e:
        print(f"Error visualizing graph: {e}")
        return None
