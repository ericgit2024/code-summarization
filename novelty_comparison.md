# Novelty Comparison Report

## Project Overview
This project implements a **Graph-Augmented Generative Code Summarization System** that explicitly fuses four distinct program representations—Abstract Syntax Tree (AST), Control Flow Graph (CFG), Program Dependence Graph (PDG), and Call Graph—into a hierarchical prompt for a Large Language Model (Gemma). The core novelty lies in the **explicit, prompt-based integration of deep structural and inter-procedural context** to generate summaries that specifically detail a function's role within the larger system (i.e., its "Called by" and "Calls" relationships), rather than just describing its internal logic.

## Comparison with Existing Research

The following table and sections compare this project against five prominent research directions in code summarization.

| Feature | **This Project** | **1. Code2Seq (Alon et al.)** | **2. GraphCodeBERT (Guo et al.)** | **3. CAST (Gong et al.)** | **4. HA-ConvGNN (Li et al.)** | **5. Standard LLM (GPT/Gemma)** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Core Representation** | **Hybrid Prompt** (AST+CFG+PDG+CG) | AST Paths | Data Flow + AST (CPG-like) | Hierarchical AST | AST + Call Graph | Raw Text / Tokens |
| **Integration Method** | **Prompt Engineering** (Textual) | Encoder Embeddings (LSTM) | Pre-training Objective (Masked) | Encoder Embeddings (Transformer) | Graph Neural Networks (GNN) | Context Window (Text) |
| **Context Scope** | **Global** (Repo-wide Call Graph) | Local (Function-level) | Local (Function-level) | Local (Function-level) | Global (Class/File) | Local (File/Snippet) |
| **Output Goal** | **Dependency-Rich Summary** | Function Name / Docstring | Representation Learning | Summary Generation | Summary Generation | General Summary |

### 1. Code2Seq (AST-Paths Approach)
**Approach:** Represents code as a set of compositional paths in the Abstract Syntax Tree (AST) and uses an encoder-decoder architecture (LSTM) to generate summaries.
**Novelty of This Project:**
- **Beyond Syntax:** Code2Seq relies solely on syntactic structure (AST). This project incorporates **semantic** (PDG) and **execution** (CFG) flows, providing a deeper understanding of *how* data moves and changes.
- **Inter-procedural Context:** Code2Seq is limited to the function's internal structure. This project explicitly retrieves and includes the **Call Graph context**, allowing the summary to explain *why* the function is called and *what* it impacts externally.

### 2. GraphCodeBERT (Pre-trained Model Approach)
**Approach:** A pre-trained model that incorporates data flow (variable sequence) alongside the source code to improve code representation learning.
**Novelty of This Project:**
- **Explicit vs. Implicit:** GraphCodeBERT learns dependencies implicitly during pre-training. This project **explicitly extracts and presents** these dependencies in the prompt, forcing the model to attend to specific relationships.
- **Generative Control:** While GraphCodeBERT is powerful for classification and clone detection, this project is tailored for **controlled generation**, ensuring the output summary follows a specific format that includes dependency information, which is not guaranteed by standard pre-trained models.

### 3. CAST (Hierarchical AST Split)
**Approach:** Addresses the issue of large ASTs by splitting them into a hierarchy of subtrees to preserve structural information without overwhelming the encoder.
**Novelty of This Project:**
- **Multi-View Fusion:** CAST focuses on optimizing the AST view. This project fuses **multiple views** (CFG, PDG, Call Graph), acknowledging that syntax alone is insufficient for summarizing complex logic.
- **Prompt-Based Fusion:** Instead of complex model architectures (custom encoders), this project leverages the **in-context learning** capabilities of modern LLMs by translating graph structures into a structured natural language prompt.

### 4. HA-ConvGNN (Hierarchical Attention GNN)
**Approach:** Uses a Graph Neural Network to combine AST and Call Graph information, propagating information from callees to callers to generate context-aware summaries.
**Novelty of This Project:**
- **Richness of Context:** While HA-ConvGNN uses the Call Graph, this project adds **PDG and CFG**, capturing fine-grained control and data dependencies that a high-level Call Graph misses.
- **Interpretability:** The intermediate step in this project involves generating a text-based "Structural Prompt." This provides a layer of **interpretability**—a user can see exactly what structural information the model is using to generate the summary, unlike the opaque embeddings of a GNN.

### 5. Standard LLM Usage (Zero-Shot/Few-Shot)
**Approach:** Feeding raw source code into models like GPT-4 or Gemma and asking for a summary.
**Novelty of This Project:**
- **Overcoming Context Blindness:** A standard LLM call typically lacks knowledge of the broader repository. This project's **RAG-like retrieval of graph neighbors** injects the necessary repository context that a standard model would miss.
- **Structured Guidance:** By explicitly parsing and presenting the CFG and PDG, this project guides the LLM to focus on **critical logic paths and data dependencies**, reducing hallucinations and generic summaries ("This function does X") in favor of precise, structural descriptions.

## Conclusion
The key innovation of this project is not just the use of graphs, but the **orchestration of multiple graph representations into a coherent, prompt-based narrative** for a generative model. It bridges the gap between **deep program analysis** (traditional static analysis) and **modern generative AI**, ensuring that the fluency of LLMs is grounded in the rigorous structural reality of the code.
