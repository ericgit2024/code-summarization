# NeuroGraph-CodeRAG: System Features & Architecture

This document provides a deep dive into the architecture and key features of **NeuroGraph-CodeRAG**, a graph-augmented agentic code summarization system.

## 1. System Architecture

The high-level architecture of NeuroGraph-CodeRAG is visualized in the `architecture.puml` file located in the root of this repository.

### Key Components:
*   **Presentation Layer (Streamlit UI):** The interactive frontend that handles user file uploads, visualizes Control Flow Graphs (CFG), and displays the generated summaries.
*   **Structural Analysis Engine:**
    *   **RepoGraphBuilder:** Parses the entire repository to build a directed graph where nodes are functions and edges represent calls. It handles cross-file dependencies and "intelligent subgraph extraction" to find relevant context.
    *   **ASTAnalyzer:** Uses Python's `ast` module to extract metadata (complexity, variables, signatures) from code.
*   **Retrieval System (RAG):** Uses `Microsoft CodeBERT` and `FAISS` to index and retrieve semantically similar code snippets to usage as few-shot examples.
*   **Reflective Agent:** A LangGraph-based agent that orchestrates the generation process.
*   **Inference Pipeline:** The backbone that fuses structural prompts, RAG context, and the Reflective Agent to drive the LLM (Gemma-2b).

---

## 2. The Reflective Agent (RL Loop)

The core innovation of this system is the **Reflective Agent**, which mimics the cognitive process of a human developer. It does not just generate text; it actively verifies and refines it.

### How it Works (The Loop)
The agent operates on a state machine built with **LangGraph**:

1.  **GENERATE:** The agent produces an initial draft summary using the available context.
2.  **CRITIQUE:** A separate "Code Reviewer" prompt analyzes the summary against the source code.
    *   *Check:* Does the summary mention `connect_db`?
    *   *Check:* Does the code call `connect_db`?
    *   *Result:* "The summary mentions database connection but doesn't explain how credentials are validated."
3.  **DECIDE (Policy Layer):** The agent decides the next action based on the critique.
    *   If "missing dependencies" are found (e.g., the code calls `validate_user` but the agent doesn't know what that function does), it transitions to **CONSULT**.
    *   If the logic is unclear but no data is missing, it transitions to **REFINE**.
    *   If the score is high, it **FINISHES**.
4.  **CONSULT:** The agent queries the **Repository Graph** for the missing functions identified in the critique. It retrieves their docstrings and signatures and appends them to the context.
5.  **REFINE:** The agent re-generates the summary, now armed with the new context about the external dependencies.

### Impact
*   **Reduced Hallucinations:** By forcing the model to verify facts against the code, the agent reduces the likelihood of inventing non-existent logic.
*   **Contextual Accuracy:** Instead of guessing what `utils.process()` does, the agent looks it up, leading to summaries like "It processes data using the sliding window algorithm defined in utils..." rather than "It processes data."

---

## 3. Repository Graph Analysis

NeuroGraph-CodeRAG moves beyond single-file analysis by building a **Repository Graph**.

*   **Node Construction:** Every function and method in the uploaded codebase is a node.
*   **Edge Creation:**
    *   **Direct Calls:** `foo()` calling `bar()` creates an edge `foo -> bar`.
    *   **Import Resolution:** The system resolves `from utils import helper` to link usages of `helper` to the actual definition in `src/utils.py`.
*   **Intelligent Subgraph Extraction:** When summarizing a function, the system doesn't just dump the whole graph. It scores neighbors based on:
    *   **Proximity:** Direct callees are higher priority.
    *   **Complexity:** Complex callees (logic-heavy) are more important than trivial getters.
    *   **Control Flow:** Dependencies inside loops or branches are boosted in importance.

---

## 4. Retrieval-Augmented Generation (RAG)

The system uses RAG to provide "few-shot" examples to the LLM, helping it understand the desired output format and tone.

*   **Encoder:** `microsoft/codebert-base` encodes code snippets into dense vectors.
*   **Index:** `FAISS` (Facebook AI Similarity Search) stores these vectors for millisecond-level retrieval.
*   **Process:**
    1.  Input code is encoded.
    2.  Top-k similar code snippets (with their existing summaries) are retrieved from the training dataset.
    3.  These examples are appended to the prompt: *"Here are examples of how to summarize code similar to this..."*

---

## 5. Structural Prompting

Instead of a flat string, the prompt sent to the LLM is a structured artifact containing:

1.  **Instruction:** "Summarize this..."
2.  **Metadata Block:** "Function: `process_data`, Complexity: 5, Args: `[data, config]`"
3.  **Dependency Context:** "This function calls `api_client.get()`, which 'retrieves data from remote server'..." (Derived from Graph)
4.  **Similar Examples:** (Derived from RAG)
5.  **Source Code:** The actual code.

This structure guides the LLM to focus on *what matters*, resulting in more technical and accurate summaries.
