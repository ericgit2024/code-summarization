# Detailed Architecture Diagram Documentation

## Overview

This document explains the comprehensive PlantUML architecture diagram for **NeuroGraph-CodeRAG**.

**File**: `detailed_architecture.puml`

---

## Diagram Structure

The diagram is organized into **6 layers**, each with detailed component breakdowns:

### **Layer 1: Presentation Layer** (Light Green #E8F5E9)

**Purpose**: User interface and visualization

**Components**:
- **Streamlit Web Interface**
  - File Upload Handler
  - Function Selector
  - Mode Toggle (Normal vs. Smart Agent)
  - Visualization Engine
    - CFG Renderer
    - Call Graph Renderer
    - AST Viewer
  - Summary Display
  - Progress Indicators

**Ports**:
- Input: User Input
- Output: Display Output

---

### **Layer 2: Application Logic Layer** (Light Blue #E3F2FD)

**Purpose**: Core workflow orchestration and agentic processing

**Components**:

#### **Inference Pipeline**
- **Orchestrator**: Coordinates entire workflow
- **Prompt Builder**: Constructs structural prompts
  - Metadata Serializer
  - CFG Serializer
  - PDG Serializer
  - Context Formatter
  - RAG Augmenter
- **Model Invoker**: Handles LLM inference
- **Response Parser**: Processes LLM output

#### **Reflective Agent (LangGraph)**
- **Agent State Manager**: Manages AgentState
  - Tracks: function_name, code, context, summary, critique, missing_deps, consulted_functions, attempts, max_attempts, action
- **LangGraph Workflow**: State machine with 5 nodes
  - **GENERATE Node**: Creates initial summary
  - **CRITIQUE Node**: Analyzes summary (score 0-10, issues, missing functions)
  - **DECIDE Node**: Decision logic (FINISH/CONSULT/REFINE)
  - **CONSULT Node**: Queries repository graph
  - **REFINE Node**: Improves summary
- **Quality Checker**: Threshold validation (8/10)

**Workflow**:
```
GENERATE → CRITIQUE → DECIDE
              ↓
    ┌────────┴────────┐
    ↓                 ↓
CONSULT           REFINE
    ↓                 ↓
    └────────┬────────┘
             ↓
        CRITIQUE (loop)
             ↓
          FINISH
```

---

### **Layer 3: Structural Analysis Layer** (Light Orange #FFF3E0)

**Purpose**: Graph construction and structural analysis

**Components**:

#### **Repository Graph Builder**
- **File Parser**: Parses all .py files
- **Function Extractor**: Extracts functions, classes, metadata
- **Call Graph Constructor**
  - Call Site Identifier
  - Import Resolver (cross-file dependencies)
  - Edge Builder
- **Subgraph Extractor**
  - Relevance Scorer: `Score = α·proximity + β·complexity + γ·cf_importance` (α=0.5, β=0.3, γ=0.2)
  - Greedy Selector: Selects within token budget
- **NetworkX Graph**: Directed graph storage
  - Nodes: Functions
  - Edges: Calls
  - Metadata: complexity, params, returns

#### **AST Analyzer**
- **Python AST Parser**: Uses Python `ast` module
- **Metadata Extractor**
  - Complexity Calculator (cyclomatic complexity)
  - Parameter Extractor
  - Variable Tracker
  - Call Identifier
- **AST Transformer**: Converts to serializable format

#### **Graph Utilities**
- **CFG Constructor**
  - Basic Block Identifier
  - Control Flow Analyzer (If/Else, While/For, Try/Except, Return)
  - Edge Labeler (True, False, Exception)
- **PDG Constructor**
  - Data Dependency Analyzer (def-use chains)
  - Control Dependency Analyzer
  - Dependency Merger
- **Graph Visualizer**
  - DOT Generator
  - Graphviz Renderer

---

### **Layer 4: Retrieval System Layer** (Light Purple #F3E5F5)

**Purpose**: RAG and similarity search

**Components**:

#### **RAG System**
- **Code Encoder**: SentenceTransformer (microsoft/codebert-base, 768-dim)
- **Index Manager**
  - Index Builder (batch_size=32)
  - Index Loader
  - Index Saver
- **Similarity Search**: Top-k retrieval (k=3)
- **Example Formatter**: Formats code-summary pairs for prompt
- **FAISS Index**: Flat L2 or IVF, dimension 768
- **Example Store**: Stores original code-summary pairs

---

### **Layer 5: Model Infrastructure Layer** (Light Pink #FCE4EC)

**Purpose**: LLM loading and inference

**Components**:

#### **Model Loader**
- **Model Initializer**: Loads google/gemma-2b from Hugging Face
- **LoRA Adapter Manager**
  - Adapter Loader
  - Adapter Merger (merges LoRA with base)
- **Device Manager**: Auto-detects CUDA/CPU
- **Tokenizer**: Gemma tokenizer, max_length=4096

#### **LLM (Gemma-2b)**
- **Base Model**: google/gemma-2b (2B parameters)
- **LoRA Adapters**
  - Rank: 8 or 16
  - Alpha: 32
  - Target: Q, V projections
  - Dropout: 0.1
- **Generation Engine**
  - temperature: 0.7
  - top_p: 0.9
  - max_length: 512
  - do_sample: True

---

### **Layer 6: Data & Training Layer** (Light Teal #E0F2F1)

**Purpose**: Training and evaluation

**Components**:

#### **Dataset Loader**
- **Custom Dataset Handler**: 386 dependency-rich examples
- **CodeXGlue Handler**: 400K+ examples from CodeSearchNet
- **Data Preprocessor**
  - Code Cleaner
  - Summary Normalizer
  - Structural Prompt Generator
- **Split Manager**: Train 80%, Val 10%, Test 10%

#### **Trainer**
- **Training Loop**
  - Batch Generator (batch_size=4, gradient_accumulation=4, effective=16)
  - Loss Computation (cross-entropy)
  - Backward Pass
- **Optimizer**: AdamW, lr=2e-4, cosine schedule
- **Checkpoint Manager**
  - Model Saver
  - Best Model Tracker (tracks best BLEU/ROUGE)
- **Metrics Logger**: Logs loss, BLEU, ROUGE, time

#### **Evaluation Metrics**
- **BLEU Calculator**: SacreBLEU (standardized)
- **ROUGE Calculator**
- **METEOR Calculator**
- **BERTScore Calculator**
- **Dependency Coverage**: `DepCov = |Deps in summary| / |Actual deps|`
- **Structural Accuracy**: Verifies control flow descriptions match CFG

#### **Databases**
- **Training Datasets**
  - code_summary_dataset.jsonl
  - codexglue_train.jsonl
  - codexglue_validation.jsonl
  - codexglue_test.jsonl
- **Model Checkpoints**
  - best_model.pt
  - checkpoint_epoch_1.pt
  - checkpoint_epoch_2.pt
  - lora_adapters.pt

---

## Data Flow

### **End-to-End Summarization Flow**

```
1. User uploads code/repository
   ↓
2. UI → File Upload Handler → Function Selector
   ↓
3. User selects mode (Normal/Smart Agent)
   ↓
4. Normal Mode:
   UI → Inference Pipeline → Orchestrator
   ↓
5. Smart Agent Mode:
   UI → Reflective Agent → State Manager
   ↓
6. Orchestrator/Agent requests:
   • Repository Graph Builder → Call Graph
   • AST Analyzer → AST + Metadata
   • Graph Utils → CFG + PDG
   ↓
7. Prompt Builder constructs structural prompt:
   • Metadata Serializer → Metadata section
   • CFG Serializer → Control flow section
   • PDG Serializer → Data dependencies section
   • Context Formatter → Repository context
   • RAG Augmenter → Similar examples
   ↓
8. Model Invoker → LLM (Gemma-2b)
   ↓
9. LLM generates summary
   ↓
10. Normal Mode: Response Parser → Summary Display
    Smart Agent Mode: CRITIQUE → DECIDE → CONSULT/REFINE (loop) → Summary Display
    ↓
11. Visualization Engine renders CFG/Call Graph
    ↓
12. User views summary + visualizations
```

---

## Key Relationships

### **Layer 1 ↔ Layer 2**
- UI sends code to Inference Pipeline (Normal mode)
- UI sends code to Reflective Agent (Smart Agent mode)
- Pipeline/Agent returns summary to UI

### **Layer 2 ↔ Layer 3**
- Orchestrator requests graph construction from RepoGraphBuilder
- Orchestrator requests AST parsing from ASTAnalyzer
- Orchestrator requests CFG/PDG from Graph Utils
- CONSULT Node queries SubgraphExtractor for missing functions
- Prompt Builder serializers pull from AST/CFG/PDG/Call Graph

### **Layer 2 ↔ Layer 4**
- RAG Augmenter retrieves similar examples from RAG System

### **Layer 2 ↔ Layer 5**
- Model Invoker sends prompts to LLM
- GENERATE/CRITIQUE/REFINE nodes invoke LLM

### **Layer 5 ↔ Layer 6**
- Trainer fine-tunes LLM with LoRA
- Model Loader loads checkpoints from Checkpoints database

### **Layer 6 Internal**
- Dataset Loader provides data to Trainer
- Trainer saves checkpoints to Checkpoints database
- Evaluation Metrics logs to Metrics Logger

---

## Component Details

### **Reflective Agent State Machine**

**AgentState** (TypedDict):
```python
{
    "function_name": str,
    "code": str,
    "context": str,
    "summary": str,
    "critique": str,
    "missing_deps": List[str],
    "consulted_functions": List[str],
    "attempts": int,
    "max_attempts": int,
    "action": str  # "CONSULT", "REFINE", or "FINISH"
}
```

**Decision Logic** (DECIDE Node):
```
IF critique.score >= 8:
    action = "FINISH"
ELIF missing_deps AND attempts < max_attempts:
    action = "CONSULT"
ELIF attempts < max_attempts:
    action = "REFINE"
ELSE:
    action = "FINISH"
```

---

### **Relevance Scoring** (Subgraph Extractor)

**Formula**:
```
Relevance(neighbor, target) = 
    α × Proximity(neighbor, target) +
    β × Complexity(neighbor) +
    γ × CFImportance(neighbor, target)
```

**Parameters**:
- α = 0.5 (proximity weight)
- β = 0.3 (complexity weight)
- γ = 0.2 (control flow importance weight)

**Proximity**:
```
Proximity = 1.0 / (shortest_path_distance + 1)
```

**Complexity**:
```
Complexity = min(cyclomatic_complexity / 10.0, 1.0)
```

---

### **Prompt Structure** (Prompt Builder)

**6 Sections**:

1. **Metadata Section**
   ```
   Function: <name>
   Complexity: <cyclomatic_complexity>
   Parameters: <params>
   Returns: <return_type>
   Local Variables: <vars>
   ```

2. **Control Flow Section**
   ```
   Control Flow:
   - Entry → <first_block>
   - <block1> → <block2> (condition: <label>)
   - ...
   ```

3. **Data Dependencies Section**
   ```
   Data Dependencies:
   - <var1> depends on: <var2>, <var3>
   - ...
   ```

4. **Repository Context Section**
   ```
   Called by:
   - <caller1> (<file>): <description>
   - ...
   
   Calls:
   - <callee1> (<file>): <description>
   - ...
   ```

5. **Similar Examples Section** (RAG)
   ```
   Similar Example 1:
   Code: <code>
   Summary: <summary>
   
   Similar Example 2:
   ...
   ```

6. **Source Code Section**
   ```python
   <actual_code>
   ```

---

## Visualization

### **How to Render the Diagram**

**Option 1: PlantUML Online**
1. Go to http://www.plantuml.com/plantuml/uml/
2. Copy contents of `detailed_architecture.puml`
3. Paste and render

**Option 2: VS Code Extension**
1. Install "PlantUML" extension
2. Open `detailed_architecture.puml`
3. Press `Alt+D` to preview

**Option 3: Command Line**
```bash
# Install PlantUML
brew install plantuml  # macOS
# or
sudo apt-get install plantuml  # Linux

# Generate PNG
plantuml detailed_architecture.puml

# Generate SVG (scalable)
plantuml -tsvg detailed_architecture.puml
```

---

## Legend

| Layer | Color | Purpose |
|:------|:------|:--------|
| Presentation | Light Green (#E8F5E9) | User Interface |
| Application Logic | Light Blue (#E3F2FD) | Core Workflow |
| Structural Analysis | Light Orange (#FFF3E0) | Graph Construction |
| Retrieval System | Light Purple (#F3E5F5) | RAG & Similarity Search |
| Model Infrastructure | Light Pink (#FCE4EC) | LLM & Inference |
| Data & Training | Light Teal (#E0F2F1) | Training & Evaluation |

---

## Key Innovations Highlighted

1. **Multi-View Structural Analysis**: AST + CFG + PDG + Call Graph (Layer 3)
2. **Reflective Agentic Workflow**: LangGraph state machine (Layer 2)
3. **Repository-Wide Context**: NetworkX call graph with cross-file resolution (Layer 3)
4. **Intelligent Subgraph Extraction**: Relevance-based selection (Layer 3)
5. **RAG Integration**: FAISS + CodeBERT for few-shot learning (Layer 4)
6. **LoRA Fine-Tuning**: Parameter-efficient adaptation (Layer 5)
7. **Custom Metrics**: Dependency coverage + structural accuracy (Layer 6)

---

## Component Count

- **Total Layers**: 6
- **Major Components**: 6 (one per layer)
- **Sub-Components**: 50+
- **Internal Components**: 80+
- **Databases/Artifacts**: 8
- **Relationships/Arrows**: 100+

---

## Usage in Thesis

This diagram can be used in:
- **Chapter 4 (Proposed Solution)**: Section 4.2 System Architecture
- **Appendix**: Full detailed architecture reference
- **Presentations**: High-level overview of system design

**Recommendation**: 
- Use simplified version (existing `architecture.puml`) in main chapter
- Reference this detailed version in appendix
- Extract specific layer diagrams for focused discussions

---

## Maintenance

When updating the system:
1. Update corresponding component in `.puml` file
2. Update relationships if data flow changes
3. Regenerate diagram
4. Update this documentation

---

**End of Documentation**
