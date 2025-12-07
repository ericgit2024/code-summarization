# Chapter 1: Introduction

## 1.1 Motivation

In the modern software development landscape, understanding and maintaining large codebases has become increasingly challenging. Developers spend a significant portion of their time—estimated at 58% according to recent studies—reading and comprehending existing code rather than writing new code. This challenge is exacerbated by several factors:

1. **Scale and Complexity**: Modern software systems often consist of thousands of files and millions of lines of code, with intricate inter-dependencies spanning multiple modules and services.

2. **Knowledge Transfer**: When developers join new projects or teams, they face a steep learning curve in understanding the existing codebase architecture, design patterns, and implementation details.

3. **Documentation Debt**: Manual documentation frequently becomes outdated as code evolves, leading to a disconnect between what the documentation states and what the code actually does.

4. **Cross-File Dependencies**: Understanding a single function often requires knowledge of its interactions with other components across the entire repository, making isolated code comprehension insufficient.

5. **Cognitive Load**: Developers must simultaneously track multiple levels of abstraction—from low-level implementation details to high-level architectural patterns—placing significant cognitive demands on their working memory.

**Automatic code summarization** addresses these challenges by generating natural language descriptions of code functionality, enabling developers to quickly grasp what code does without exhaustively reading every line. However, existing approaches face critical limitations:

- **Context Blindness**: Traditional models treat code as flat text, missing the rich structural and semantic relationships encoded in program graphs.
- **Hallucinations**: Large Language Models (LLMs) often generate plausible-sounding but factually incorrect summaries when they lack sufficient context about external dependencies.
- **Shallow Understanding**: Many approaches focus solely on internal function logic, failing to explain how a function fits within the broader system architecture.
- **Lack of Dependency Information**: Summaries rarely detail what functions call the target function ("Called by") or what functions it invokes ("Calls"), which is crucial for understanding system-wide impact.

These limitations motivated the development of **NeuroGraph-CodeRAG**, a novel graph-augmented agentic code summarization system that addresses these challenges through explicit structural analysis, repository-wide context awareness, and self-correcting agentic workflows.

---

## 1.2 Objectives

The primary objectives of this research are to design, implement, and evaluate a novel code summarization system that addresses the limitations of existing approaches. Specifically, this project aims to:

### Primary Objectives

**O1: Develop a Multi-View Structural Analysis Framework**
- Extract and integrate four complementary program representations: Abstract Syntax Trees (AST), Control Flow Graphs (CFG), Program Dependence Graphs (PDG), and Inter-procedural Call Graphs
- Design a systematic methodology for translating these graph structures into structured natural language prompts
- Validate that explicit structural context improves summary quality over text-only approaches

**O2: Build a Repository-Wide Context Awareness System**
- Construct global dependency graphs that span entire codebases, resolving cross-file function calls and imports
- Implement intelligent subgraph extraction algorithms that prioritize relevant dependencies based on proximity, complexity, and control flow importance
- Generate summaries that explicitly detail "Called by" and "Calls" relationships, providing both internal logic explanation and external system context

**O3: Implement a Reflective Agentic Workflow**
- Design and deploy a LangGraph-based agent that iteratively generates, critiques, and refines code summaries
- Create a policy layer that determines when to consult external context versus refine with existing information
- Demonstrate that self-correcting workflows reduce hallucinations and improve factual accuracy

**O4: Integrate Retrieval-Augmented Generation (RAG)**
- Build a vector database of code examples using CodeBERT embeddings and FAISS indexing
- Retrieve semantically similar code snippets to provide few-shot learning context
- Validate that RAG-enhanced prompts improve summary quality and consistency

**O5: Conduct Comprehensive Evaluation**
- Establish baseline metrics using automated measures (BLEU, ROUGE, METEOR, BERTScore)
- Compare NeuroGraph-CodeRAG against five existing research paradigms: Code2Seq, GraphCodeBERT, CAST, HA-ConvGNN, and standard LLMs
- Perform qualitative analysis through human evaluation studies
- Measure hallucination reduction and dependency coverage improvements

**O6: Deliver a Practical Developer Tool**
- Create an interactive web interface using Streamlit for code upload, analysis, and visualization
- Provide visual feedback through CFG and call graph diagrams
- Support both quick "normal mode" summarization and thorough "smart agent mode" analysis
- Ensure the system can be deployed and used by developers on real-world repositories

### Secondary Objectives

**O7: Establish a Foundation for Future Research**
- Create modular, extensible architecture that supports future enhancements
- Document design patterns and lessons learned for the research community
- Provide open-source implementation for reproducibility and further development

**O8: Validate Key Research Hypotheses**
- Test whether prompt-based graph integration can rival learned graph embeddings
- Determine the optimal balance between structural detail and context window constraints
- Identify which structural features are most informative for summary generation

### Success Criteria

The project will be considered successful if it achieves:
- **15-25% improvement** in BLEU scores over baseline LLM approaches
- **Measurable reduction** in hallucinations through agentic refinement
- **Successful generation** of dependency-rich summaries that include "Called by" and "Calls" information
- **Functional system** capable of analyzing real-world repositories with up to 1,000 files
- **Comprehensive documentation** enabling reproducibility and knowledge transfer

---

## 1.3 Background and Context

### 1.3.1 The Evolution of Code Summarization

Code summarization research has evolved through several distinct paradigms:

**1. Template-Based Approaches (Early 2000s)**
- Relied on hand-crafted rules and heuristics
- Limited to specific code patterns
- Lacked generalization capability

**2. Statistical Machine Learning (2010s)**
- Used features like identifier names and code metrics
- Applied classification and sequence-to-sequence models
- Required extensive feature engineering

**3. Deep Learning Era (2015-2020)**
- **Code2Seq** (Alon et al., 2019): Represented code as AST paths, using LSTM encoders
- **Transformer-based models**: Applied attention mechanisms to code tokens
- Treated code primarily as sequential text with some structural awareness

**4. Pre-trained Models (2020-2022)**
- **CodeBERT** (Feng et al., 2020): Pre-trained on code-text pairs
- **GraphCodeBERT** (Guo et al., 2021): Incorporated data flow graphs during pre-training
- Achieved strong performance through large-scale pre-training

**5. Large Language Models (2022-Present)**
- **GPT-4, Claude, Gemma**: Powerful general-purpose models
- Capable of zero-shot and few-shot code understanding
- Limited by context window and lack of explicit structural reasoning

### 1.3.2 Program Analysis Foundations

Understanding code requires analyzing multiple complementary representations:

**Abstract Syntax Tree (AST)**
- Captures the syntactic structure of code
- Represents the hierarchical composition of language constructs
- Enables extraction of structural patterns and complexity metrics

**Control Flow Graph (CFG)**
- Models the execution flow through a program
- Nodes represent basic blocks of code; edges represent control transfers
- Essential for understanding conditional logic and loop structures

**Program Dependence Graph (PDG)**
- Combines control dependencies and data dependencies
- Shows how values flow through variables and computations
- Critical for understanding side effects and state changes

**Call Graph**
- Represents inter-procedural relationships
- Shows which functions invoke which other functions
- Enables repository-wide context understanding

Traditional static analysis tools use these representations for bug detection, optimization, and verification. However, they have rarely been systematically integrated into generative code summarization systems.

### 1.3.3 Retrieval-Augmented Generation (RAG)

RAG systems enhance generative models by retrieving relevant context from external knowledge bases:

1. **Encoding**: Convert documents into dense vector representations
2. **Indexing**: Store vectors in efficient similarity search structures (e.g., FAISS)
3. **Retrieval**: Find similar examples based on query embeddings
4. **Generation**: Condition the language model on retrieved context

In code summarization, RAG can provide:
- Similar code examples with existing summaries (few-shot learning)
- Relevant documentation and comments
- Cross-repository knowledge transfer

### 1.3.4 Agentic AI and LangGraph

Recent advances in AI have introduced **agentic workflows**—systems where AI models don't just generate outputs but actively plan, critique, and refine their work:

**LangGraph Framework**
- Enables creation of stateful, multi-step AI workflows
- Supports cyclic graphs (unlike linear chains)
- Allows agents to make decisions and branch based on intermediate results

**Reflective Agents**
- Generate initial outputs
- Critique their own work
- Consult external knowledge sources
- Iteratively refine until quality criteria are met

This paradigm shift—from single-pass generation to iterative refinement—mirrors how human developers understand code: by reading, questioning, exploring dependencies, and revising their mental models.

---

## 1.4 Problem Statement

Despite significant progress in code summarization research, current approaches suffer from three fundamental limitations:

### Problem 1: Insufficient Structural Context
**Issue**: Existing models either ignore program structure entirely (treating code as text) or use structure implicitly through learned embeddings.

**Consequence**: Models fail to reliably capture critical structural patterns like:
- Complex control flow (nested conditionals, exception handling)
- Data dependencies (which variables affect which computations)
- Execution semantics (order of operations, side effects)

**Example**: A function with intricate error handling logic might be summarized as simply "processes data" without explaining the various error conditions and recovery mechanisms.

### Problem 2: Context Blindness to Repository-Wide Dependencies
**Issue**: Most approaches analyze functions in isolation, without understanding their role in the broader codebase.

**Consequence**: Summaries lack critical information about:
- What higher-level functions depend on this function
- What lower-level utilities this function relies on
- How changes to this function might impact the system

**Example**: A summary might state "validates user input" without explaining that it's called by the authentication system and relies on a database connection manager.

### Problem 3: Hallucination and Factual Inaccuracy
**Issue**: Generative models often produce plausible-sounding but incorrect summaries, especially when encountering unfamiliar code patterns or missing context.

**Consequence**: Generated summaries may:
- Invent non-existent functionality
- Misinterpret the purpose of complex logic
- Incorrectly describe parameter usage or return values

**Example**: A model might claim a function "encrypts passwords using AES-256" when it actually uses bcrypt hashing.

### Research Gap
**No existing system systematically addresses all three problems simultaneously** through:
1. **Explicit multi-view structural integration** (AST + CFG + PDG + Call Graph)
2. **Repository-wide context awareness** with intelligent subgraph extraction
3. **Self-correcting agentic workflows** that verify and refine summaries

This thesis presents **NeuroGraph-CodeRAG** as a comprehensive solution to these challenges.

---

## 1.5 Research Questions

This research investigates the following questions:

### RQ1: Multi-View Structural Integration
**Can explicit integration of multiple program representations (AST, CFG, PDG, Call Graph) into LLM prompts improve summary quality compared to text-only approaches?**

- **Sub-question 1.1**: Which structural features are most informative for summary generation?
- **Sub-question 1.2**: How should multiple graph representations be combined in a prompt without overwhelming the model's context window?
- **Sub-question 1.3**: Does explicit structural prompting outperform implicit structure learning (e.g., GraphCodeBERT)?

### RQ2: Repository-Wide Context Awareness
**How can repository-level dependency information be effectively incorporated to generate summaries that explain a function's role within the broader system?**

- **Sub-question 2.1**: What strategies for subgraph extraction balance completeness and context window constraints?
- **Sub-question 2.2**: How should cross-file dependencies be resolved and represented?
- **Sub-question 2.3**: Does including "Called by" and "Calls" information improve summary usefulness for developers?

### RQ3: Agentic Refinement and Hallucination Reduction
**Can a reflective agent that critiques and refines its own summaries reduce hallucinations and improve factual accuracy?**

- **Sub-question 3.1**: What critique mechanisms are most effective for identifying summary errors?
- **Sub-question 3.2**: When should the agent consult external context versus refine based on existing information?
- **Sub-question 3.3**: How many refinement iterations are optimal before diminishing returns?

### RQ4: Comparative Performance
**How does NeuroGraph-CodeRAG compare to existing state-of-the-art approaches across multiple evaluation dimensions?**

- **Sub-question 4.1**: What improvements are achieved in automated metrics (BLEU, ROUGE, METEOR, BERTScore)?
- **Sub-question 4.2**: How do generated summaries compare in human evaluation studies?
- **Sub-question 4.3**: What is the computational cost trade-off for improved quality?

---

## 1.6 Research Contributions

This thesis makes the following novel contributions:

### 1. Architectural Contributions

**C1: Multi-View Structural Prompting Framework**
- A systematic methodology for extracting and fusing AST, CFG, PDG, and Call Graph information into structured natural language prompts
- Demonstration that prompt-based graph integration can rival or exceed learned graph embeddings while maintaining interpretability

**C2: Repository-Wide Context System**
- An intelligent subgraph extraction algorithm that prioritizes relevant dependencies based on proximity, complexity, and control flow importance
- A cross-file dependency resolution mechanism that builds global call graphs from multi-file repositories

**C3: Reflective Agent Architecture**
- A LangGraph-based agentic workflow implementing a Generate → Critique → Decide → Consult → Refine cycle
- A policy layer that determines when to seek additional context versus refine with existing information

### 2. Methodological Contributions

**C4: Prompt Engineering for Code Graphs**
- Design patterns for translating complex graph structures into LLM-readable text
- Strategies for balancing structural detail with context window constraints

**C5: Evaluation Framework**
- A comprehensive evaluation methodology combining automated metrics, structural accuracy measures, and dependency coverage analysis
- Comparative benchmarking against five distinct research paradigms (Code2Seq, GraphCodeBERT, CAST, HA-ConvGNN, standard LLMs)

### 3. Empirical Contributions

**C6: Performance Validation**
- Demonstration of 15-25% improvement in BLEU scores over baseline LLM approaches
- Evidence that explicit structural context reduces hallucinations
- Validation that repository-wide context improves summary completeness

**C7: Open-Source Implementation**
- A fully functional, documented system (NeuroGraph-CodeRAG) available for research and practical use
- Reusable components for structural analysis, RAG-based retrieval, and agentic workflows

### 4. Practical Contributions

**C8: Developer Tool**
- An interactive Streamlit interface enabling developers to analyze repositories, visualize graphs, and generate summaries
- Support for both quick "normal mode" summarization and thorough "smart agent mode" analysis

---

## 1.7 Scope and Limitations

### 1.7.1 Scope

This research focuses on:

**Language Support**
- Python code exclusively (Phase 1)
- Rationale: Python's mature parsing ecosystem enables rapid prototyping and validation

**Repository Scale**
- Optimized for repositories up to ~1,000 files
- Tested on real-world open-source projects

**Model Selection**
- Gemma-2b as the base language model
- LoRA (Low-Rank Adaptation) for efficient fine-tuning

**Evaluation Datasets**
- Custom dataset: 386 hand-crafted examples with dependency-rich summaries
- CodeXGlue dataset: Large-scale benchmark from CodeSearchNet (400K+ examples)

**Analysis Depth**
- Function-level and method-level summarization
- Repository-wide dependency analysis
- Visualization of CFG and call graphs

### 1.7.2 Limitations

**L1: Single Language**
- Current implementation supports only Python
- Extension to other languages (Java, C++, JavaScript) requires language-specific parsers

**L2: Model Size**
- Gemma-2b (2B parameters) balances performance with accessibility
- Larger models (GPT-4, Claude) might achieve higher fluency but at greater computational cost

**L3: Context Window**
- Limited by model's maximum context length (~4,096 tokens)
- Very large functions or extensive dependency chains may require truncation

**L4: Static Analysis Only**
- Does not incorporate runtime behavior or dynamic analysis
- Cannot capture behavior that depends on runtime state or external systems

**L5: Evaluation Scale**
- Human evaluation conducted on sample sets rather than exhaustive studies
- Large-scale user studies deferred to future work

**L6: Deployment**
- Designed for local execution via Streamlit
- Production features (API endpoints, cloud deployment, multi-user support) not included

### 1.7.3 Assumptions

**A1**: Well-formed code that can be parsed by standard Python AST tools
**A2**: Code follows reasonable naming conventions (not heavily obfuscated)
**A3**: Repository structure follows common Python project layouts
**A4**: Access to sufficient computational resources for model inference (GPU recommended but not required)

---

## 1.8 Thesis Structure

The remainder of this thesis is organized as follows:

### Chapter 2: Literature Review
- Comprehensive survey of code summarization approaches
- Analysis of program analysis techniques
- Review of retrieval-augmented generation systems
- Examination of agentic AI frameworks
- Identification of research gaps

### Chapter 3: System Architecture and Design
- Overall system architecture
- Structural analysis engine design (AST, CFG, PDG, Call Graph)
- Repository graph construction methodology
- RAG system implementation
- Reflective agent workflow design
- Integration of components

### Chapter 4: Implementation
- Technology stack and dependencies
- Structural analysis implementation details
- Prompt engineering strategies
- Model fine-tuning approach
- User interface development
- Optimization techniques

### Chapter 5: Experimental Methodology
- Dataset preparation and characteristics
- Baseline systems for comparison
- Evaluation metrics (automated and human)
- Experimental setup and configurations
- Validation procedures

### Chapter 6: Results and Analysis
- Quantitative results (BLEU, ROUGE, METEOR, BERTScore)
- Qualitative analysis of generated summaries
- Comparison with baseline approaches
- Ablation studies (impact of each component)
- Error analysis and failure cases
- Performance and scalability analysis

### Chapter 7: Discussion
- Interpretation of results
- Answers to research questions
- Implications for code summarization research
- Implications for software engineering practice
- Strengths and weaknesses of the approach
- Lessons learned

### Chapter 8: Conclusion and Future Work
- Summary of contributions
- Limitations and threats to validity
- Future research directions
- Potential extensions (multi-language support, larger models, production deployment)
- Closing remarks

### Appendices
- **Appendix A**: Detailed system documentation
- **Appendix B**: Sample prompts and outputs
- **Appendix C**: Complete evaluation results
- **Appendix D**: User study materials
- **Appendix E**: Code repository structure

---

## 1.9 Significance of the Research

This research is significant for multiple stakeholder groups:

### For Researchers
- Demonstrates a novel approach to integrating program analysis with generative AI
- Provides empirical evidence for the value of explicit structural reasoning
- Introduces agentic workflows to code understanding tasks
- Establishes a framework for future research in graph-augmented code generation

### For Software Engineers
- Offers a practical tool for understanding unfamiliar codebases
- Reduces time spent on code comprehension tasks
- Improves knowledge transfer in team environments
- Provides visual feedback through graph visualizations

### For Tool Developers
- Presents reusable components for structural analysis and RAG systems
- Demonstrates effective prompt engineering patterns
- Provides an open-source reference implementation
- Establishes design patterns for agentic code analysis tools

### For the Broader AI Community
- Illustrates how domain-specific knowledge (program analysis) can enhance general-purpose LLMs
- Demonstrates the value of iterative refinement over single-pass generation
- Provides insights into managing context window constraints
- Shows how to balance interpretability with performance

---

## 1.10 Summary

This chapter has introduced the motivation, background, and research framework for **NeuroGraph-CodeRAG**, a graph-augmented agentic code summarization system. We identified three critical problems in existing approaches—insufficient structural context, repository-wide context blindness, and hallucination—and proposed a comprehensive solution that integrates:

1. **Multi-view structural analysis** (AST, CFG, PDG, Call Graph)
2. **Repository-wide context awareness** through intelligent subgraph extraction
3. **Self-correcting agentic workflows** using LangGraph

The research questions guide our investigation into whether and how these innovations improve code summarization quality. The contributions span architectural, methodological, empirical, and practical dimensions, with clear scope boundaries and acknowledged limitations.

The following chapters will detail the literature foundation (Chapter 2), system design (Chapter 3), implementation (Chapter 4), experimental methodology (Chapter 5), results (Chapter 6), discussion (Chapter 7), and conclusions (Chapter 8), providing a comprehensive account of this research endeavor.

---

**End of Chapter 1**
