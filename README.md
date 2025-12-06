# NeuroGraph-CodeRAG: Graph-Augmented Agentic Code Summarization

## Overview

**NeuroGraph-CodeRAG** (formerly SP-RAG) is a state-of-the-art code summarization system that fuses **Static Analysis**, **Graph Theory**, and **Generative AI**. Unlike traditional models that treat code as flat text, NeuroGraph-CodeRAG constructs a multi-layered understanding of the codebase by extracting and combining four distinct graph structures: **Abstract Syntax Trees (AST)**, **Control Flow Graphs (CFG)**, **Program Dependence Graphs (PDG)**, and **Inter-procedural Call Graphs**.

The system features an advanced **Reflective Agent** powered by **LangGraph**, which iteratively critiques and refines its own summaries by autonomously consulting the repository graph for missing context.

For a deep dive into the system's architecture and features, see [FEATURES.md](./FEATURES.md).

![Architecture Diagram](./architecture.puml)
*Note: The architecture diagram is available in PlantUML format as `architecture.puml` in the repository root.*

## Key Features

1.  **Multi-View Structural Prompting:** Integrates AST, CFG, PDG, and Call Graphs into a single, comprehensive prompt.
2.  **Repo-Wide Context Awareness:** Resolves cross-file dependencies to understand interactions across the entire codebase.
3.  **Reflective Agent (LangGraph):** An agentic workflow that Generates, Critiques, Decides, Consults, and Refines to ensure accuracy and reduce hallucinations.
4.  **Retrieval-Augmented Generation (RAG):** Retrieves similar code examples from a FAISS vector database to guide the model.

[Read more about these features in FEATURES.md](./FEATURES.md)

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

### 5. Dataset Selection

The system supports two datasets:

**Custom Dataset** (default):
- 386 hand-crafted examples with dependency-rich summaries
- Located in `code_summary_dataset.jsonl`

**CodeXGlue Dataset**:
- Large-scale dataset from CodeSearchNet (400K+ Python examples)
- Requires download and preprocessing (see [CODEXGLUE_INTEGRATION.md](./CODEXGLUE_INTEGRATION.md))

**Automated Pipeline (Recommended)**:
```bash
# Single command to download, preprocess, split, build RAG, and train
python run_codexglue_pipeline.py --subset 10000 --epochs 5
```

**Manual Setup** (if you prefer step-by-step):
```bash
# Download subset (recommended for testing)
python -m src.scripts.download_codexglue --subset 10000 --output codexglue_raw.jsonl --validate

# Preprocess with structural features
python -m src.scripts.preprocess_codexglue --input codexglue_raw.jsonl --output codexglue_processed.jsonl

# Create train/val/test splits
python -m src.scripts.create_dataset_splits --input codexglue_processed.jsonl
```

### 6. Training (Optional)
To fine-tune the model on your chosen dataset:

**Custom dataset**:
```bash
python3 -m src.model.trainer
```

**CodeXGlue dataset**:
```bash
python3 -m src.model.trainer --dataset-name codexglue
```

For detailed CodeXGlue integration instructions, see [CODEXGLUE_INTEGRATION.md](./CODEXGLUE_INTEGRATION.md).

## How the Reflective Agent Works

The **Reflective Agent** follows a cognitive cycle implemented with **LangGraph**:
1.  **Generate:** Produces a draft summary.
2.  **Critique:** A separate LLM call evaluates the summary against the code.
3.  **Policy Check:** If missing dependencies are found, the agent decides to **Consult**.
4.  **Consult:** The agent queries the **Repo Graph** for the missing function.
5.  **Refine:** The agent rewrites the summary with the new context.

*For a detailed explanation of this loop, please refer to [FEATURES.md](./FEATURES.md).*

## Project Structure
*   `src/structure`: Graph algorithms (AST, CFG, Repo Graph).
*   `src/model`: LLM inference, LoRA training, and the `ReflectiveAgent`.
*   `src/ui`: Streamlit application.
*   `src/retrieval`: RAG system implementation.
