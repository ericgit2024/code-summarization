import streamlit as st
import sys
import os
import shutil
import networkx as nx
import tempfile

# Add the project root to sys.path to resolve 'src' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.model.inference import InferencePipeline
from src.utils.repo_analysis import RepoAnalyzer
from src.ui.visualization import visualize_dependency_graph, visualize_call_graph
from src.structure.graph_utils import visualize_cfg

st.set_page_config(page_title="SP-RAG Code Summarizer", layout="wide")

st.title("SP-RAG Code Summarization System")
st.markdown("Generates structurally accurate and semantically rich code summaries using Gemma (4-bit), RAG, and Structural Prompting.")

# Initialize pipeline
@st.cache_resource
def load_pipeline():
    return InferencePipeline()

pipeline = load_pipeline()

# Initialize session state
if 'python_files' not in st.session_state:
    st.session_state.python_files = []
if 'dependencies' not in st.session_state:
    st.session_state.dependencies = None
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'call_graph' not in st.session_state:
    st.session_state.call_graph = None
if 'repo_handler' not in st.session_state:
    st.session_state.repo_handler = None  # Store TemporaryDirectory object
if 'repo_dir' not in st.session_state:
    st.session_state.repo_dir = None

def cleanup_repo():
    if st.session_state.repo_handler:
        st.session_state.repo_handler.cleanup()
        st.session_state.repo_handler = None
        st.session_state.repo_dir = None
        print("Cleaned up temporary repository.")

# Input Section
st.header("Input")
repo_url = st.text_input("GitHub Repository URL", "https://github.com/psf/requests")

if st.button("Analyze Repository"):
    if repo_url:
        with st.spinner("Cloning and Analyzing Repository..."):
            cleanup_repo()
            # Create a new temporary directory
            temp_dir = tempfile.TemporaryDirectory(prefix="repo_")
            st.session_state.repo_handler = temp_dir
            st.session_state.repo_dir = temp_dir.name

            st.session_state.analyzer = RepoAnalyzer(repo_url, target_dir=st.session_state.repo_dir)
            st.session_state.analyzer.clone_repo()
            st.session_state.python_files = st.session_state.analyzer.find_python_files()
            st.session_state.dependencies = st.session_state.analyzer.get_dependencies()
            st.session_state.call_graph = st.session_state.analyzer.build_call_graph()

            # Analyze and print graph metrics
            if st.session_state.dependencies:
                dep_graph = nx.DiGraph()
                for file, imports in st.session_state.dependencies.items():
                    dep_graph.add_node(file)
                    for imp in imports:
                        dep_graph.add_edge(file, imp)
                print("\n--- Dependency Graph Metrics ---")
                print(st.session_state.analyzer.analyze_graph_metrics(dep_graph))

            if st.session_state.call_graph:
                print("\n--- Call Graph Metrics ---")
                print(st.session_state.analyzer.analyze_graph_metrics(st.session_state.call_graph))

            st.success("Repository analysis complete.")
    else:
        st.error("Please provide a Repo URL.")

if st.session_state.python_files:
    summarization_target = st.selectbox("Summarization Target", ["Function", "Class", "File"])
    selected_file = st.selectbox("Select a Python file", st.session_state.python_files)

    if summarization_target == "Function":
        target_name = st.text_input("Target Function Name")
    elif summarization_target == "Class":
        target_name = st.text_input("Target Class Name")
    else:
        target_name = None

    if st.button("Get Code"):
        if selected_file and st.session_state.analyzer:
            code = None
            if summarization_target == "Function":
                if target_name:
                    code = st.session_state.analyzer.extract_function_code(selected_file, target_name)
                else:
                    st.error("Please enter a function name.")
            elif summarization_target == "Class":
                if target_name:
                    code = st.session_state.analyzer.extract_class_code(selected_file, target_name)
                else:
                    st.error("Please enter a class name.")
            elif summarization_target == "File":
                code = st.session_state.analyzer.extract_file_code(selected_file)

            if code:
                st.session_state.code_input = code
            else:
                st.error(f"{summarization_target} not found.")
        else:
            st.error("Please select a file.")

code_input = st.text_area("Code to Summarize", height=300, key="code_input")


# Process
if st.button("Generate Summary"):
    if code_input:
        with st.spinner("Generating summary..."):
            summary = pipeline.summarize(code_input)
            st.success("Summary Generated!")
            st.subheader("Summary")
            st.write(summary)

            st.subheader("Structural Analysis")
            cfg_graph = visualize_cfg(code_input)
            if cfg_graph:
                st.graphviz_chart(cfg_graph)
            else:
                st.error("Could not generate CFG.")

    else:
        st.error("Please provide code to summarize.")

if st.session_state.dependencies:
    st.subheader("Dependency Graph")
    dep_graph_viz = visualize_dependency_graph(st.session_state.dependencies)
    if dep_graph_viz:
        st.graphviz_chart(dep_graph_viz)
    else:
        st.error("Could not generate dependency graph.")

if st.session_state.call_graph:
    st.subheader("Call Graph")
    call_graph_viz = visualize_call_graph(st.session_state.call_graph)
    if call_graph_viz:
        st.graphviz_chart(call_graph_viz)
    else:
        st.error("Could not generate call graph.")


st.sidebar.header("About")
st.sidebar.info(
    "This system uses:\n"
    "- **Gemma-2b/7b (4-bit)** for generation\n"
    "- **RAG** for context retrieval\n"
    "- **AST + CFG** for structural prompting\n"
)
