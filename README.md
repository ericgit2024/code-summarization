# SP-RAG: Structurally-Aware Code Summarization

## Overview

SP-RAG (Structural Prompting with Retrieval-Augmented Generation) is a research project designed to generate high-quality, context-aware summaries of source code. Traditional code summarization models often struggle to capture the nuances of code's structural properties, leading to summaries that are either too generic or semantically disconnected from the code's actual execution flow.

This project introduces a novel approach that integrates **Structural Prompting**—using Abstract Syntax Trees (ASTs), Control Flow Graphs (CFGs), and Inter-procedural Call Graphs—with a **Retrieval-Augmented Generation (RAG)** pipeline. By doing so, SP-RAG provides the language model with a richer, more holistic understanding of the code, resulting in summaries that are not only fluent but also structurally and semantically precise.

## Research Perspective

From a research standpoint, SP-RAG explores the intersection of several key areas in software engineering and natural language processing:

1.  **Code as a Modality:** We treat source code not as plain text, but as a structured modality. By extracting and linearizing structural information (ASTs, CFGs, Call Graphs), we aim to create a more effective representation for large language models.
2.  **Structural Prompting:** This project investigates the efficacy of injecting explicit structural information into prompts. The hypothesis is that by providing the model with a "blueprint" of the code's architecture and logic, it can generate more accurate and faithful summaries.
3.  **Inter-procedural Analysis:** The system analyzes the call graph of an entire repository. This allows the model to generate summaries for higher-level components (like classes or modules) by understanding the interactions and dependencies between different functions, leading to more holistic documentation.
4.  **Retrieval-Augmented Generation for Code:** While RAG has proven effective for knowledge-intensive NLP tasks, its application to code summarization is still an emerging area. SP-RAG explores how to best retrieve relevant code snippets and structural information to augment the generation process.
5.  **Efficient Fine-Tuning:** The project leverages techniques like QLoRA (Quantized Low-Rank Adaptation) to efficiently fine-tune large language models (like Gemma) on the code summarization task, making it feasible to train powerful models on consumer-grade hardware.

## How It Works

The SP-RAG pipeline consists of the following stages:

1.  **Structural Feature Extraction:** Given a piece of code or a repository, we first extract its AST, CFG, and call graph. These structures represent the code's syntactic, control-flow, and inter-procedural properties, respectively.
2.  **Structural Prompt Construction:** The extracted features are serialized into a textual format and integrated into a "structural prompt." This prompt provides the language model with a multi-faceted view of the code.
3.  **Retrieval-Augmented Generation:** The structural prompt is used to query a vector database of code snippets and their corresponding summaries. The retrieved examples are then added to the prompt as few-shot exemplars, providing the model with relevant context.
4.  **Code Summarization:** The final, augmented prompt is fed to the language model (e.g., Gemma), which generates the code summary.

## Usage

To use the SP-RAG system, you can either paste a code snippet directly into the UI or provide a link to a GitHub repository.

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

-   **Full PDG Integration:** The current implementation uses a placeholder for PDG extraction. Future work will involve integrating a robust PDG analysis tool to capture data dependencies more effectively.
-   **Expanded Language Support:** The current system is primarily focused on Python. Future iterations will aim to support other programming languages.
-   **Evaluation on Larger Benchmarks:** The model will be evaluated on a wider range of code summarization benchmarks to assess its performance against state-of-the-art methods.
-   **IDE Integration:** We envision SP-RAG as a tool that can be integrated directly into IDEs, providing developers with real-time code summarization and documentation assistance.
