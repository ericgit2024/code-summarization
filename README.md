# NeuroGraph-CodeRAG: Graph-Augmented Agentic Code Summarization

## Overview

**NeuroGraph-CodeRAG** (formerly SP-RAG) is a state-of-the-art code summarization system that fuses **Static Analysis**, **Graph Theory**, and **Generative AI**. Unlike traditional models that treat code as flat text, NeuroGraph-CodeRAG constructs a multi-layered understanding of the codebase by extracting and combining four distinct graph structures: **Abstract Syntax Trees (AST)**, **Control Flow Graphs (CFG)**, **Program Dependence Graphs (PDG)**, and **Inter-procedural Call Graphs**.

The system features an advanced **Reflective Agent** powered by **LangGraph**, which iteratively critiques and refines its own summaries by autonomously consulting the repository graph for missing context (e.g., unexplained function calls).

## Key Features

1.  **Multi-View Structural Prompting:** Integrates AST (syntax), CFG (execution flow), PDG (data dependencies), and Call Graphs (inter-procedural context) into a single, comprehensive prompt.
2.  **Repo-Wide Context Awareness:** Builds a graph of the entire repository to resolve cross-file dependencies. It understands that a function calling `save_user()` depends on the database module, even if that code isn't in the current file.
3.  **Reflective Agent (LangGraph):** An agentic workflow that:
    *   **Generates** an initial summary.
    *   **Critiques** it for missing details or hallucinations.
    *   **Consults** the repository graph to fetch definitions of unknown functions.
    *   **Refines** the summary based on new evidence.
4.  **Retrieval-Augmented Generation (RAG):** Retrieves similar code examples from a vector database (FAISS) to guide the model via few-shot learning.

## How the Graphs Work Together

The system combines four graph representations, prioritizing them based on their utility for summarization:

1.  **Instruction & Metadata (AST):** The prompt starts with high-level metadata derived from the **AST** (function signature, parameters, cyclomatic complexity). This sets the stage.
2.  **Dependency Context (Call Graph):** The **Call Graph** is the most critical for context. The system injects a textual description of *relevant* dependencies (callees), explaining what they do (based on their docstrings) and why they are relevant (relevance scoring).
3.  **Logic & Flow (CFG/PDG):** While not always dumped as raw text, the **CFG** drives the complexity analysis (identifying loops/branches) and is visualized in the UI to help the user understand the code's "shape".
4.  **Source Code:** Finally, the raw source code provides the minute details.

**Prompt Structure:** `Instruction > AST Metadata > Call Graph Context > RAG Examples > Source Code`

## Usage Guide

### 1. Prerequisites
*   Python 3.8+
*   Git
*   Graphviz (`sudo apt-get install graphviz` or `brew install graphviz`)
*   Hugging Face Account & Token (for Gemma model)

### 2. Installation
```bash
git clone https://github.com/yourusername/NeuroGraph-CodeRAG.git
cd NeuroGraph-CodeRAG
pip install -r requirements.txt
```

### 3. Setup Authentication
Export your Hugging Face token:
*   **Linux/Mac:** `export HF_TOKEN="your_token_here"`
*   **Windows:** `$env:HF_TOKEN="your_token_here"`

### 4. Build the RAG Index
Initialize the vector database of code examples:
```bash
python3 -m src.scripts.build_rag_index
```

### 5. Running the Interface
Launch the interactive web UI:
```bash
python3 -m streamlit run src/ui/app.py
```
*   **Analyze Repo Dump:** Upload a combined `.py` file containing your repository's code. The system will build the full graph.
*   **Debug & Visualize:** Select a function to see its **Control Flow Graph (CFG)** visualized and its **Repository Context** (dependencies) listed.
*   **Smart Agent:** Check "Use Smart Agent" to enable the self-correcting workflow.

### 6. Training (Optional)
To fine-tune the model on your own dataset:
```bash
python3 -m src.model.trainer
```

## How the Reflective Agent Works

The **Reflective Agent** follows a cognitive cycle implemented with **LangGraph**:
1.  **Generate:** Produces a draft summary.
2.  **Critique:** A separate LLM call evaluates the summary against the code. *Does it mention `connect_db` but fail to explain it?*
3.  **Policy Check:** If the critique finds "missing dependencies," the agent decides to **Consult**.
4.  **Consult:** The agent queries the **Repo Graph** for the missing function (`connect_db`), retrieves its docstring/signature, and adds it to the context.
5.  **Refine:** The agent rewrites the summary with the new context.

This mimics a human developer reading code, realizing they don't know a function, looking it up, and then writing a better explanation.

## Project Structure
*   `src/structure`: Graph algorithms (AST, CFG, Repo Graph).
*   `src/model`: LLM inference, LoRA training, and the `ReflectiveAgent`.
*   `src/ui`: Streamlit application.
*   `src/retrieval`: RAG system implementation.
