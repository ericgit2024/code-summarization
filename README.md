# SP-RAG: Structurally-Aware Code Summarization

## Overview

SP-RAG (Structural Prompting with Retrieval-Augmented Generation) is a research project designed to generate high-quality, context-aware summaries of source code. Traditional code summarization models often struggle to capture the nuances of code's structural properties, leading to summaries that are either too generic or semantically disconnected from the code's actual execution flow.

This project introduces a novel approach that integrates **Structural Prompting**—using Abstract Syntax Trees (ASTs), Control Flow Graphs (CFGs), and Program Dependence Graphs (PDGs)—with a **Retrieval-Augmented Generation (RAG)** pipeline. By doing so, SP-RAG provides the language model with a richer, more holistic understanding of the code, resulting in summaries that are not only fluent but also structurally and semantically precise.

## Research Perspective

From a research standpoint, SP-RAG explores the intersection of several key areas in software engineering and natural language processing:

1.  **Code as a Modality:** We treat source code not as plain text, but as a structured modality. By extracting and linearizing structural information (ASTs, CFGs, PDGs, Call Graphs), we aim to create a more effective representation for large language models.
2.  **Structural Prompting:** This project investigates the efficacy of injecting explicit structural information into prompts. The hypothesis is that by providing the model with a "blueprint" of the code's architecture and logic, it can generate more accurate and faithful summaries.
3.  **Inter-procedural Analysis:** The system implements a name-based resolution strategy to build a call graph of the repository. This allows the model to generate summaries that consider the interactions and dependencies between different functions, moving beyond isolated function-level analysis.
4.  **Retrieval-Augmented Generation for Code:** SP-RAG utilizes a vector database (FAISS) with `sentence-transformers` to retrieve relevant code snippets. These retrieved examples serve as few-shot exemplars in the prompt, providing the model with similar code contexts to guide generation.
5.  **Efficient Fine-Tuning:** The project leverages **QLoRA** (Quantized Low-Rank Adaptation) to efficiently fine-tune the **Gemma-2b** model. This approach enables training on consumer-grade hardware (4-bit quantization) while maintaining high performance.
6.  **Evaluation Metrics:** The system includes a robust evaluation pipeline supporting standard n-gram and embedding-based metrics: **BLEU**, **ROUGE**, and **METEOR**.

## How It Works

The SP-RAG pipeline consists of the following stages:

1.  **Structural Feature Extraction:** Given a piece of code, we extract its AST, CFG, and PDG. For repositories, we build a dependency graph to capture inter-procedural relationships.
2.  **Structural Prompt Construction:** The extracted features are serialized into a textual format (e.g., linearizing the AST and CFG nodes) and integrated into a "structural prompt."
3.  **Retrieval-Augmented Generation:** The input code is embedded and used to query a FAISS vector index containing a database of code-summary pairs. Top-k similar examples are retrieved and added to the prompt context.
4.  **Code Summarization:** The final prompt—combining the structural blueprint, retrieved exemplars, and the target code—is fed to the fine-tuned Gemma model to generate the summary.

## Project Structure

-   `src/model`: Contains model loading (Gemma + LoRA), training logic, and evaluation scripts.
-   `src/structure`: Implements AST parsing, CFG generation (`py2cfg`), and PDG extraction (Control/Data dependencies).
-   `src/retrieval`: Implements the RAG system using FAISS and SentenceTransformers.
-   `src/utils`: Contains repository analysis tools for dependency graph construction.
-   `src/ui`: A Streamlit-based user interface for interacting with the system.
-   `src/data`: Handles dataset loading (custom JSONL format) and preprocessing.

## Usage

### Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Set up your Hugging Face token (required for Gemma):
    ```bash
    export HF_TOKEN='your_token_here'  # Linux/Mac
    $env:HF_TOKEN='your_token_here'    # Windows PowerShell
    ```

### Running the UI

To use the SP-RAG system, you can run the Streamlit interface:

```bash
streamlit run src/ui/app.py
```

### From a GitHub Repository

1.  Enter the URL of the GitHub repository.
2.  Click "Analyze Repository." The UI will display the dependency graph and the inter-procedural call graph.
3.  Select a Python file from the dropdown menu.
4.  Enter the name of the function you want to summarize.
5.  Click "Get Function Code."
6.  Click "Generate Summary."

### From a Code Snippet

1.  Paste your code into the text area.
2.  Click "Generate Summary."

## Future Work

-   **Expanded Language Support:** The current system is primarily focused on Python. Future iterations will aim to support other programming languages.
-   **Evaluation on Larger Benchmarks:** The model will be evaluated on standard benchmarks (e.g., CodeSearchNet) to assess its performance against state-of-the-art methods.
-   **IDE Integration:** We envision SP-RAG as a tool that can be integrated directly into IDEs.
