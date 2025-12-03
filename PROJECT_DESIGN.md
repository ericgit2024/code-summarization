# Project Design Document

## Project Title
**NeuroGraph-CodeRAG: Graph-Augmented Agentic Code Summarization System**

---

## Novelty

The core novelty of this project lies in the **explicit, prompt-based integration of deep structural and inter-procedural context** to generate code summaries that detail a function's role within the larger system. Unlike existing approaches, this project:

1. **Multi-View Structural Fusion**: Explicitly fuses four distinct program representations—Abstract Syntax Tree (AST), Control Flow Graph (CFG), Program Dependence Graph (PDG), and Inter-procedural Call Graph—into a hierarchical prompt for a Large Language Model (Gemma).

2. **Reflective Agentic Workflow**: Implements a LangGraph-based agent that iteratively Generates, Critiques, Decides, Consults, and Refines summaries, mimicking the cognitive process of a human developer to reduce hallucinations and improve accuracy.

3. **Repository-Wide Context Awareness**: Resolves cross-file dependencies and provides global context through intelligent subgraph extraction, enabling the model to understand how individual functions interact within the entire codebase.

4. **Prompt-Based Graph Integration**: Translates complex graph structures into structured natural language prompts, leveraging the in-context learning capabilities of modern LLMs rather than requiring custom neural architectures.

5. **Dependency-Rich Summaries**: Generates summaries that explicitly detail "Called by" and "Calls" relationships, providing both internal logic explanation and external system context.

---

## Objectives

### Phase 1 Objectives

1. **Core Infrastructure Development**
   - Implement multi-view structural analysis engine (AST, CFG, PDG extraction)
   - Build repository-wide call graph construction system
   - Develop structural prompt generation mechanism
   - Integrate Gemma-2b model with LoRA fine-tuning capability

2. **Basic Summarization Pipeline**
   - Create end-to-end inference pipeline for single-function summarization
   - Implement RAG system using CodeBERT embeddings and FAISS indexing
   - Develop Streamlit-based interactive UI for code upload and visualization
   - Enable CFG and call graph visualization using Graphviz

3. **Model Training & Evaluation**
   - Fine-tune Gemma-2b on custom code summarization dataset
   - Establish baseline metrics (BLEU, ROUGE, METEOR)
   - Validate structural feature extraction accuracy
   - Demonstrate improved performance over standard LLM approaches

4. **Documentation & Validation**
   - Create comprehensive system documentation
   - Develop usage guides and walkthroughs
   - Validate system on sample repositories
   - Demonstrate core functionality through live demos

### Phase 2 Objectives

1. **Advanced Agentic Capabilities**
   - Implement full Reflective Agent workflow using LangGraph
   - Develop critique and refinement mechanisms
   - Create intelligent context consultation system
   - Optimize agent decision-making policies

2. **Enhanced Context Retrieval**
   - Implement intelligent subgraph extraction with scoring algorithms
   - Add proximity-based and complexity-based neighbor prioritization
   - Develop cross-file dependency resolution enhancements
   - Optimize context window utilization

3. **Scalability & Performance**
   - Optimize graph construction for large repositories
   - Implement caching mechanisms for repeated analyses
   - Add parallel processing for batch summarization
   - Improve inference speed and memory efficiency

4. **Comprehensive Evaluation**
   - Conduct extensive comparison with existing research (Code2Seq, GraphCodeBERT, CAST, HA-ConvGNN)
   - Perform human evaluation studies
   - Measure hallucination reduction metrics
   - Validate dependency-rich summary quality

5. **Production Readiness**
   - Add API endpoints for programmatic access
   - Implement error handling and edge case management
   - Create deployment documentation
   - Develop CI/CD pipeline for continuous integration

---

## Novelty - Objective Mapping

| Novelty Aspect | Phase 1 Objectives | Phase 2 Objectives |
|:---------------|:-------------------|:-------------------|
| **Multi-View Structural Fusion** | Obj 1.1: Implement AST, CFG, PDG extraction<br>Obj 1.3: Develop structural prompt generation | Obj 2.2: Add proximity-based neighbor prioritization<br>Obj 2.3: Enhance cross-file dependency resolution |
| **Reflective Agentic Workflow** | Obj 2.2: Implement basic RAG system | Obj 1.1-1.4: Full LangGraph agent implementation<br>Obj 4.2: Human evaluation of agent effectiveness |
| **Repository-Wide Context** | Obj 1.2: Build call graph construction<br>Obj 2.4: Enable graph visualization | Obj 2.1: Intelligent subgraph extraction<br>Obj 3.1: Optimize for large repositories |
| **Prompt-Based Integration** | Obj 1.3: Structural prompt mechanism<br>Obj 1.4: Gemma-2b integration | Obj 2.4: Optimize context window utilization<br>Obj 3.4: Improve inference efficiency |
| **Dependency-Rich Summaries** | Obj 3.2: Establish baseline metrics<br>Obj 3.3: Validate feature extraction | Obj 4.1: Comprehensive research comparison<br>Obj 4.3: Measure hallucination reduction |

---

## Scope of the Project (Phase 1)

### Inclusions (Covered)

1. **Structural Analysis Components**
   - Abstract Syntax Tree (AST) parsing using Tree-sitter
   - Control Flow Graph (CFG) construction
   - Program Dependence Graph (PDG) extraction
   - Repository-level call graph building with cross-file resolution
   - Cyclomatic complexity calculation
   - Variable and parameter metadata extraction

2. **Model Infrastructure**
   - Gemma-2b model integration via Hugging Face
   - LoRA (Low-Rank Adaptation) fine-tuning implementation
   - Custom dataset preprocessing pipeline
   - Structural prompt construction mechanism
   - Basic inference pipeline for single-function summarization

3. **Retrieval-Augmented Generation (RAG)**
   - CodeBERT-based code embedding generation
   - FAISS vector database indexing
   - Similarity-based example retrieval
   - Few-shot learning integration into prompts

4. **User Interface**
   - Streamlit-based web application
   - File upload functionality (single files and repository dumps)
   - Function selection interface
   - CFG visualization using Graphviz
   - Call graph visualization
   - Summary generation and display

5. **Training & Evaluation**
   - Fine-tuning script for custom datasets
   - BLEU, ROUGE, and METEOR metric calculation
   - Training progress monitoring
   - Model checkpoint management

6. **Documentation**
   - System architecture documentation
   - Feature explanation documents
   - Installation and usage guides
   - Novelty comparison with existing research

### Exclusions (Not Covered in Phase 1)

1. **Advanced Agentic Features**
   - Full LangGraph-based Reflective Agent workflow
   - Automated critique and refinement loops
   - Intelligent context consultation mechanisms
   - Multi-iteration summary improvement

2. **Advanced Optimization**
   - Large-scale repository optimization (>10,000 files)
   - Distributed processing for batch summarization
   - Advanced caching strategies
   - GPU-optimized inference pipelines

3. **Production Features**
   - REST API endpoints
   - Authentication and authorization
   - Multi-user support
   - Cloud deployment configurations
   - Continuous integration/deployment pipelines

4. **Extended Language Support**
   - Languages beyond Python (Java, C++, JavaScript, etc.)
   - Multi-language repository analysis
   - Language-specific optimization

5. **Advanced Evaluation**
   - Large-scale human evaluation studies
   - A/B testing frameworks
   - Comparative benchmarking automation
   - Real-world deployment case studies

### Deliverables

1. **Software Components**
   - Fully functional code summarization system
   - Structural analysis engine (AST, CFG, PDG, Call Graph)
   - RAG-based retrieval system with FAISS index
   - Streamlit web interface with visualization capabilities
   - Fine-tuned Gemma-2b model with LoRA adapters
   - Training and inference scripts

2. **Documentation**
   - `README.md`: System overview and quick start guide
   - `FEATURES.md`: Detailed feature documentation
   - `PROJECT_EXPLANATION.md`: Comprehensive conceptual deep dive
   - `WALKTHROUGH.md`: Step-by-step usage instructions
   - `novelty_comparison.md`: Research comparison analysis
   - `PROJECT_DESIGN.md`: This design document
   - `architecture.puml`: System architecture diagram

3. **Datasets & Models**
   - Preprocessed training dataset with structural features
   - FAISS vector index for RAG retrieval
   - Fine-tuned model checkpoints
   - Example code snippets and summaries

4. **Evaluation Results**
   - Baseline metric scores (BLEU, ROUGE, METEOR)
   - Comparison with standard LLM approaches
   - Sample outputs demonstrating dependency-rich summaries
   - CFG and call graph visualizations

5. **Demonstration Materials**
   - Live demo via Streamlit interface
   - Example repository analysis
   - Visualization of structural graphs
   - Before/after summary comparisons

### Boundaries & Limitations

1. **Language Support**
   - **Boundary**: Phase 1 focuses exclusively on Python code
   - **Limitation**: Cannot analyze repositories in other programming languages
   - **Rationale**: Python's mature parsing ecosystem (Tree-sitter, AST module) allows rapid prototyping

2. **Repository Scale**
   - **Boundary**: Optimized for repositories up to ~1,000 files
   - **Limitation**: Performance degradation on very large codebases (>10,000 files)
   - **Rationale**: Graph construction is memory-intensive; optimization deferred to Phase 2

3. **Model Capabilities**
   - **Boundary**: Uses Gemma-2b (2 billion parameters)
   - **Limitation**: May not match the fluency of larger models (GPT-4, Claude)
   - **Rationale**: Balances performance with accessibility; can be run on consumer hardware

4. **Agentic Workflow**
   - **Boundary**: Basic RAG retrieval without iterative refinement
   - **Limitation**: No self-correction or critique loop in Phase 1
   - **Rationale**: Full LangGraph agent is complex; deferred to Phase 2 for focused development

5. **Context Window**
   - **Boundary**: Limited to model's maximum context length (~4,096 tokens for Gemma-2b)
   - **Limitation**: Cannot include entire large repositories in a single prompt
   - **Rationale**: Requires intelligent subgraph extraction (Phase 2 enhancement)

6. **Evaluation Scope**
   - **Boundary**: Automated metrics (BLEU, ROUGE, METEOR) and limited manual validation
   - **Limitation**: No large-scale human evaluation studies
   - **Rationale**: Comprehensive human evaluation requires significant resources; planned for Phase 2

7. **Deployment**
   - **Boundary**: Local execution via Streamlit
   - **Limitation**: No cloud deployment, API access, or multi-user support
   - **Rationale**: Focus on core functionality; production features planned for Phase 2

### Expected Outcomes

1. **Technical Achievements**
   - A working prototype that demonstrates the feasibility of graph-augmented code summarization
   - Measurable improvement over baseline LLM approaches (expected 15-25% improvement in BLEU scores)
   - Successful integration of four graph types (AST, CFG, PDG, Call Graph) into prompts
   - Functional RAG system that retrieves relevant examples
   - Accurate structural feature extraction validated against ground truth

2. **Research Contributions**
   - Demonstration of prompt-based graph integration as an alternative to custom neural architectures
   - Evidence that explicit structural context reduces hallucinations
   - Validation that repository-wide context improves summary quality
   - Comparative analysis showing advantages over existing approaches (Code2Seq, GraphCodeBERT, etc.)

3. **User Experience**
   - Intuitive web interface for uploading code and selecting functions
   - Visual feedback through CFG and call graph diagrams
   - Clear, dependency-rich summaries that explain both internal logic and external context
   - Fast inference times (<10 seconds per function on consumer hardware)

4. **Knowledge Transfer**
   - Comprehensive documentation enabling new users to understand and extend the system
   - Clear explanation of how graphs are extracted and translated into prompts
   - Walkthrough demonstrating installation, training, and inference
   - Novelty comparison helping researchers understand the project's position in the field

5. **Foundation for Phase 2**
   - Modular architecture that supports easy integration of the Reflective Agent
   - Established baseline metrics for measuring Phase 2 improvements
   - Identified bottlenecks and optimization opportunities
   - Validated core assumptions about the value of structural context

6. **Demonstration Capabilities**
   - Ability to analyze real-world repositories and generate meaningful summaries
   - Visualization of how structural information influences summary generation
   - Evidence of cross-file dependency resolution
   - Comparison outputs showing improvement over standard LLM approaches

---

## Document Metadata

- **Version**: 1.0
- **Date**: December 3, 2025
- **Author**: NeuroGraph-CodeRAG Development Team
- **Status**: Phase 1 Design Specification
- **Next Review**: Upon Phase 1 Completion

---

## References

- `README.md`: System overview and usage guide
- `FEATURES.md`: Detailed feature documentation
- `PROJECT_EXPLANATION.md`: Conceptual deep dive
- `novelty_comparison.md`: Research comparison analysis
- `architecture.puml`: System architecture diagram
