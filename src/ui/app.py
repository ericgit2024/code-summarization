import streamlit as st
import sys
import os

# Add the project root to sys.path to resolve 'src' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.model.inference import InferencePipeline
from src.utils.repo_analysis import RepoAnalyzer
from src.ui.visualization import visualize_dependency_graph
from src.structure.graph_utils import visualize_cfg
import os

st.set_page_config(page_title="SP-RAG Code Summarizer", layout="wide")

st.title("SP-RAG Code Summarization System")
st.markdown("Generates structurally accurate and semantically rich code summaries using Gemma (4-bit), RAG, and Structural Prompting.")

# Initialize pipeline
@st.cache_resource
def load_pipeline():
    return InferencePipeline()

pipeline = load_pipeline()

# Input Section
st.header("Input")
repo_url = st.text_input("GitHub Repository URL", "https://github.com/psf/requests")
func_name = st.text_input("Target Function Name (Optional, for demo purposes enters code directly)")
code_input = st.text_area("Or Paste Code Here", height=300)

# Process
if st.button("Generate Summary"):
    if code_input:
        with st.spinner("Generating summary..."):
            summary = pipeline.summarize(code_input)
            st.success("Summary Generated!")
            st.subheader("Summary")
            st.write(summary)

            # Visualize Dependency (Mock for single code snippet)
            # For full repo, we would use RepoAnalyzer
            st.subheader("Structural Analysis")
            # Mock dependency graph for the single function
            # st.write("Dependency graph visualization would appear here for full repository analysis.")
            
            cfg_graph = visualize_cfg(code_input)
            if cfg_graph:
                st.graphviz_chart(cfg_graph)
            else:
                st.error("Could not generate CFG.")

    elif repo_url:
        with st.spinner("Cloning and Analyzing Repository..."):
            analyzer = RepoAnalyzer(repo_url)
            # analyzer.clone_repo() # Disabled for demo security/performance in some envs
            # dependencies = analyzer.get_dependencies()

            st.warning("Full repository analysis is disabled in this demo environment to prevent heavy network/disk usage. Please paste code above.")

            # if dependencies:
            #     st.subheader("Dependency Graph")
            #     img_path = visualize_dependency_graph(dependencies)
            #     if img_path and os.path.exists(img_path):
            #         st.image(img_path)

    else:
        st.error("Please provide a Repo URL or Paste Code.")

st.sidebar.header("About")
st.sidebar.info(
    "This system uses:\n"
    "- **Gemma-2b/7b (4-bit)** for generation\n"
    "- **RAG** for context retrieval\n"
    "- **AST + CFG** for structural prompting\n"
)
