import streamlit as st
import sys
import os

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


# Input Section
st.header("Input")
repo_url = st.text_input("GitHub Repository URL", "https://github.com/psf/requests")

if st.button("Analyze Repository"):
    if repo_url:
        with st.spinner("Cloning and Analyzing Repository..."):
            st.session_state.analyzer = RepoAnalyzer(repo_url)
            st.session_state.analyzer.clone_repo()
            st.session_state.python_files = st.session_state.analyzer.find_python_files()
            st.session_state.dependencies = st.session_state.analyzer.get_dependencies()
            st.session_state.call_graph = st.session_state.analyzer.build_call_graph()
            st.success("Repository analysis complete.")
    else:
        st.error("Please provide a Repo URL.")

if st.session_state.python_files:
    selected_file = st.selectbox("Select a Python file", st.session_state.python_files)
    func_name = st.text_input("Target Function Name")

    if st.button("Get Function Code"):
        if selected_file and func_name and st.session_state.analyzer:
            code = st.session_state.analyzer.extract_function_code(selected_file, func_name)
            if code:
                st.session_state.code_input = code
            else:
                st.error(f"Function '{func_name}' not found in '{selected_file}'.")
        else:
            st.error("Please select a file and enter a function name.")

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
    img_path = visualize_dependency_graph(st.session_state.dependencies)
    if img_path and os.path.exists(img_path):
        st.image(img_path)
    else:
        st.warning("Could not visualize the dependency graph. Ensure graphviz is installed.")

if st.session_state.call_graph:
    st.subheader("Call Graph")
    img_path = visualize_call_graph(st.session_state.call_graph)
    if img_path and os.path.exists(img_path):
        st.image(img_path)
    else:
        st.warning("Could not visualize the call graph. Ensure graphviz is installed.")


st.sidebar.header("About")
st.sidebar.info(
    "This system uses:\n"
    "- **Gemma-2b/7b (4-bit)** for generation\n"
    "- **RAG** for context retrieval\n"
    "- **AST + CFG** for structural prompting\n"
)
