# NeuroGraph-CodeRAG: Implementation & Presentation Phasing

This document outlines the strategic division of the **NeuroGraph-CodeRAG** project into two distinct phases for presentation. This approach allows you to demonstrate a working "Core" system first (Phase 1), followed by the advanced "Agentic" capabilities (Phase 2).

---

## Phase 1: The Foundation (Structural RAG)
**Theme:** "Enhancing LLMs with Static Analysis and Retrieval"

**Objective:** Demonstrate that injecting strict program structure (AST/CFG) and similar code examples (RAG) into the prompt significantly improves summary quality compared to a raw LLM.

### 1. Scope of Presentation
*   **Input:** Single function code.
*   **Core Logic:**
    *   **Static Analysis:** Extraction of AST (Syntax), CFG (Control Flow), and PDG (Data Dependencies).
    *   **RAG:** Retrieval of similar code snippets using CodeBERT + FAISS.
    *   **Prompt Engineering:** Constructing the "Structural Prompt" (Code + Graph Metadata + Retrieved Examples).
*   **Model:** Gemma-2b (Fine-tuned or Few-shot).
*   **UI:** Basic Streamlit interface to upload a file and view the generated summary.

### 2. Key Deliverables to Show
*   **Visuals:**
    *   Show the raw code vs. the "Structural Prompt" (to prove you aren't just sending raw text).
    *   Visualize one CFG (using the Graphviz output).
*   **Demo Flow:**
    1.  User inputs a function.
    2.  System extracts "Complexity: 5", "Calls: [A, B]", "Variables: [x, y]".
    3.  System retrieves 2 similar examples.
    4.  Gemma generates a summary.

### 3. Novelty Highlight (for Phase 1)
*   **vs. Standard LLMs:** You are not blindly trusting the model. You are *grounding* it with extracted facts (CFG/PDG).
*   **vs. GraphCodeBERT:** You are using these graphs *explicitly* in the prompt, making the process interpretable.

---

## Phase 2: The Evolution (Agentic & Repo-Aware)
**Theme:** "From Static Summarization to Reflective Agents"

**Objective:** Address the limitations of Phase 1 (hallucinations, missing external context) by introducing the **Reflective Agent** and **Repository Graph**.

### 1. Scope of Presentation
*   **Input:** Entire Repository (Zip/Folder).
*   **Core Logic:**
    *   **Repo Graph:** Building the global dependency graph (Function A calls Function B in File C).
    *   **LangGraph Agent:** The `Generate -> Critique -> Consult -> Refine` loop.
    *   **Self-Correction:** Show the agent realizing it doesn't know what a function does, and *actively looking it up*.
*   **Evaluation:** Quantitative metrics (BLEU, ROUGE, BERTScore).

### 2. Key Deliverables to Show
*   **The "Aha!" Moment:**
    *   Show a "Draft Summary" (Phase 1 style) that guesses what a helper function does.
    *   Show the **Critique**: "Error: `utils.encrypt` is called but not explained."
    *   Show the **Consult**: Agent queries the Repo Graph for `utils.encrypt`.
    *   Show the **Final Summary**: Correctly explains `utils.encrypt`.
*   **Metrics:** Present the table comparing Phase 1 (Base) vs. Phase 2 (Agentic) scores.

### 3. Gap Analysis & Future Integrations (Critical for Phase 2)
To make Phase 2 a complete "Thesis-level" project, you must address these gaps:

#### A. Missing Integrations (To Implement)
1.  **Robust Critique Prompts:**
    *   *Current State:* The critique might be too generic.
    *   *Plan:* Tune the "Reviewer" prompt to specifically check for *hallucinated variable names* or *missing return value descriptions*.
2.  **Graph Visualization in UI:**
    *   *Current State:* You can see the CFG of one function.
    *   *Plan:* Add a view to see the **Repo Subgraph** (e.g., a central node with arrows pointing to its dependencies). This proves "Global Context".
3.  **A/B Testing Toggle:**
    *   *Current State:* Checkbox for "Smart Agent".
    *   *Plan:* Ensure the UI clearly displays *both* the "Fast Summary" (Phase 1) and "Smart Summary" (Phase 2) side-by-side for immediate comparison.

#### B. Evaluation Plan (Must Have)
You cannot just say it's better; you must prove it.
*   **Dataset:** Use your `code_summary_dataset.jsonl`. Split into Train/Test.
*   **Baselines:**
    1.  **Gemma-2b (Raw):** Just the code.
    2.  **Phase 1 (Structural):** Code + CFG/AST.
    3.  **Phase 2 (Agentic):** The full pipeline.
*   **Metrics:**
    *   **BLEU/ROUGE:** For textual overlap.
    *   **BERTScore:** For semantic similarity.
    *   **Human Eval (Optional but strong):** Manually check 10 summaries for "Hallucinations" (Did it invent a fact?).

---

## Summary of Novelty (The "Why This Matters" Slide)

| Feature | Phase 1 (Foundation) | Phase 2 (Agentic) |
| :--- | :--- | :--- |
| **Context** | Local (Function only) | Global (Repo-wide dependencies) |
| **Logic** | Linear (Input -> Output) | Cyclic (Generate <-> Critique) |
| **Reliability** | High (for syntax), Med (for logic) | Very High (Self-corrects errors) |
| **Analogy** | A Junior Dev reading one file. | A Senior Dev exploring the whole codebase. |
