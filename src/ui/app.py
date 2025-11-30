import streamlit as st
import sys
import os
import tempfile

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.model.inference import InferencePipeline
from src.structure.graph_utils import visualize_cfg

st.set_page_config(page_title="SP-RAG Code Summarizer", layout="wide")

st.title("SP-RAG Code Summarization System")
st.markdown("Generates context-aware code summaries using Repo-Level Graph Analysis.")

# Initialize pipeline (Model is loaded once)
@st.cache_resource
def load_pipeline():
    return InferencePipeline()

try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Sidebar for Mode Selection
mode = st.sidebar.radio("Input Mode", ["Upload Repo Dump"])

if mode == "Upload Repo Dump":
    st.header("Upload Repository Dump")
    st.info("Upload a single .py file containing the concatenated code of the repository.")
    
    uploaded_file = st.file_uploader("Choose a .py file", type="py")
    
    if uploaded_file is not None:
        # Save uploaded file to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        st.success(f"File uploaded: {uploaded_file.name}")
        
        # Build Graph
        with st.spinner("Building Repository Graph..."):
            pipeline.build_repo_graph(tmp_file_path)
        st.success("Graph built successfully!")
        
        target_func = st.text_input("Target Function Name", placeholder="e.g., main")
        
        use_smart_agent = st.checkbox("Use Smart Agent (LangGraph)", value=False, help="Enable iterative refinement using LangGraph.")

        if st.button("Generate Summary"):
            if target_func:
                try:
                    with st.spinner(f"Generating summary for '{target_func}'..."):
                        if use_smart_agent:
                            summary = pipeline.summarize_with_agent(function_name=target_func)
                            st.success("Smart Summary Generated!")
                        else:
                            summary = pipeline.summarize(function_name=target_func)
                        
                        st.subheader("Generated Summary")
                        st.write(summary)
                        
                        # Show Context
                        st.subheader("Repository Context")
                        context = pipeline.repo_graph.get_context_text(target_func)
                        st.text(context)
                        
                        # Show Code
                        st.subheader("Function Code")
                        node_data = pipeline.repo_graph.graph.nodes.get(target_func)
                        if node_data:
                            code = node_data.get("code", "")
                            st.code(code, language="python")
                            
                            # CFG Visualization
                            st.subheader("Control Flow Graph")
                            cfg = visualize_cfg(code)
                            if cfg:
                                st.graphviz_chart(cfg)
                        else:
                            st.warning("Function code not found in graph nodes.")
                            
                except Exception as e:
                    st.error(f"Error during summarization: {e}")
            else:
                st.warning("Please enter a target function name.")
