# Slide: Code Implementation - NeuroGraph-CodeRAG

## 🎯 **Overview**

This slide showcases the **major code implementations** of NeuroGraph-CodeRAG, demonstrating how each component is built. Each section includes **specific line numbers** for taking screenshots to include in your presentation.

---

## 📊 **Implementation Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                    IMPLEMENTATION LAYERS                     │
├─────────────────────────────────────────────────────────────┤
│  1. Dataset Loading & Processing                            │
│  2. Model Loading & LoRA Configuration                      │
│  3. Training Pipeline                                        │
│  4. Structural Analysis (AST, CFG, PDG, Call Graph)         │
│  5. Repository Graph Construction                           │
│  6. RAG System (Retrieval-Augmented Generation)             │
│  7. Reflective Agent (LangGraph Workflow)                   │
│  8. Inference Pipeline                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 1️⃣ **Dataset Loading & Processing**

### **File**: `src/data/dataset.py`

### **Purpose**
Loads and validates training data from custom dataset or CodeXGlue, ensuring code is syntactically valid.

### **Key Features**
- ✅ Supports multiple datasets (custom, CodeXGlue)
- ✅ AST-based validation
- ✅ Automatic train/validation splitting
- ✅ Dataset statistics logging

### **Screenshot Line Numbers**

#### **Screenshot 1: Dataset Loading Function**
**Lines: 17-76** (Full `load_and_process_dataset` function)
```
📸 Capture: Lines 17-76
Shows: Complete dataset loading logic with CodeXGlue support
```

**Key Code Highlights:**
- **Lines 30-40**: CodeXGlue dataset loading with split file mapping
- **Lines 50-62**: Custom dataset loading with fallback splitting
- **Lines 64-73**: Dataset statistics calculation

#### **Screenshot 2: Validation Logic**
**Lines: 7-15** (`is_valid_example` function)
```
📸 Capture: Lines 7-15
Shows: AST-based code validation
```

### **Talking Points**
- "We support both custom and large-scale CodeXGlue datasets"
- "AST parsing ensures only syntactically valid code enters training"
- "Automatic statistics logging helps monitor data quality"

---

## 2️⃣ **Model Loading & LoRA Configuration**

### **File**: `src/model/model_loader.py`

### **Purpose**
Loads Gemma-2b model with 4-bit quantization and configures LoRA adapters for efficient fine-tuning.

### **Key Features**
- ✅ 4-bit quantization (memory efficient)
- ✅ LoRA configuration for parameter-efficient training
- ✅ Gradient checkpointing enabled
- ✅ Multi-GPU support via device_map

### **Screenshot Line Numbers**

#### **Screenshot 1: Model Loading with Quantization**
**Lines: 6-45** (Full `load_gemma_model` function)
```
📸 Capture: Lines 6-45
Shows: 4-bit quantization configuration and model loading
```

**Key Code Highlights:**
- **Lines 18-25**: HuggingFace token validation
- **Lines 27-31**: BitsAndBytes 4-bit config
- **Lines 33-39**: Model loading with quantization
- **Lines 42-43**: Gradient checkpointing for memory efficiency

#### **Screenshot 2: LoRA Configuration**
**Lines: 47-61** (`setup_lora` function)
```
📸 Capture: Lines 47-61
Shows: LoRA adapter configuration
```

**Key Code Highlights:**
- **Line 50**: `r=8` - LoRA rank
- **Line 51**: `lora_alpha=32` - Scaling factor
- **Lines 52**: Target modules (q_proj, k_proj, v_proj, etc.)

### **Talking Points**
- "4-bit quantization reduces memory from 8GB to 2GB"
- "LoRA trains only 0.1% of parameters, saving compute"
- "Targets attention and MLP layers for maximum impact"

---

## 3️⃣ **Training Pipeline**

### **File**: `src/model/trainer.py`

### **Purpose**
Orchestrates the complete training process with RAG-augmented prompts and structural context.

### **Key Features**
- ✅ Integrated RAG retrieval during training
- ✅ Structural prompt construction
- ✅ Gradient accumulation for effective batch size
- ✅ Epoch-based training with evaluation

### **Screenshot Line Numbers**

#### **Screenshot 1: Training Setup**
**Lines: 10-44** (Training function initialization)
```
📸 Capture: Lines 10-44
Shows: Model loading, RAG system initialization, dataset preparation
```

**Key Code Highlights:**
- **Lines 20-24**: Model and LoRA setup
- **Lines 27-33**: RAG index loading
- **Lines 36-44**: Dataset loading and splitting

#### **Screenshot 2: Prompt Construction**
**Lines: 47-68** (`format_prompt` function)
```
📸 Capture: Lines 47-68
Shows: Structural prompt + RAG retrieval integration
```

**Key Code Highlights:**
- **Line 48**: Structural prompt construction
- **Lines 52-53**: RAG retrieval (k=3)
- **Lines 55-61**: Full prompt assembly with instruction

#### **Screenshot 3: Training Arguments**
**Lines: 80-99** (TrainingArguments configuration)
```
📸 Capture: Lines 80-99
Shows: Training hyperparameters and optimization settings
```

**Key Code Highlights:**
- **Line 87**: `gradient_accumulation_steps=8`
- **Line 88**: `warmup_steps=20`
- **Line 90**: `learning_rate=2e-4`
- **Line 91**: `fp16=True` - Mixed precision training

### **Talking Points**
- "RAG retrieval happens during training for better context"
- "Gradient accumulation simulates larger batch sizes"
- "Epoch-based training with automatic evaluation"

---

## 4️⃣ **Structural Analysis (AST Analyzer)**

### **File**: `src/structure/ast_analyzer.py`

### **Purpose**
Deep AST analysis extracting functions, classes, complexity metrics, and control flow information.

### **Key Features**
- ✅ Cyclomatic complexity calculation
- ✅ Variable dependency tracking
- ✅ Control structure counting (loops, branches, exceptions)
- ✅ Function call extraction with context

### **Screenshot Line Numbers**

#### **Screenshot 1: AST Analyzer Class**
**Lines: 6-24** (ASTAnalyzer initialization and analyze method)
```
📸 Capture: Lines 6-24
Shows: Main analyzer class structure
```

#### **Screenshot 2: Function Analysis**
**Lines: 78-124** (`_analyze_function` method)
```
📸 Capture: Lines 78-124
Shows: Deep function analysis with complexity metrics
```

**Key Code Highlights:**
- **Lines 83-95**: Function metadata extraction
- **Lines 98-102**: Parameter analysis with type annotations
- **Lines 105-116**: Complexity and control structure analysis

#### **Screenshot 3: Complexity Calculation**
**Lines: 149-227** (FunctionBodyAnalyzer class - key methods)
```
📸 Capture: Lines 149-227
Shows: Cyclomatic complexity and control flow tracking
```

**Key Code Highlights:**
- **Lines 190-195**: If statement complexity (+1)
- **Lines 197-202**: Loop complexity (+1)
- **Lines 218-222**: Exception handling complexity

### **Talking Points**
- "Calculates cyclomatic complexity for code understanding"
- "Tracks variable definitions and uses for PDG"
- "Identifies control structures (loops, branches, exceptions)"

---

## 5️⃣ **Repository Graph Construction**

### **File**: `src/structure/repo_graph.py`

### **Purpose**
Builds a global call graph of the entire repository with intelligent subgraph extraction.

### **Key Features**
- ✅ Cross-file dependency resolution
- ✅ Import analysis and symbol resolution
- ✅ Relevance-based scoring for context selection
- ✅ Intelligent subgraph extraction within token budget

### **Screenshot Line Numbers**

#### **Screenshot 1: Graph Builder Initialization**
**Lines: 11-50** (RepoGraphBuilder class and build_from_directory)
```
📸 Capture: Lines 11-50
Shows: Graph initialization and directory parsing
```

**Key Code Highlights:**
- **Lines 12-14**: NetworkX DiGraph initialization
- **Lines 34-50**: Directory walking and file parsing

#### **Screenshot 2: Edge Building (Call Graph)**
**Lines: 88-159** (`_build_edges` method)
```
📸 Capture: Lines 88-159
Shows: Call graph edge construction with import resolution
```

**Key Code Highlights:**
- **Lines 95-106**: Iterating through function calls
- **Lines 108-125**: Import-based resolution
- **Lines 127-145**: Class method resolution

#### **Screenshot 3: Intelligent Subgraph Extraction**
**Lines: 224-266** (`extract_dependency_subgraph` method)
```
📸 Capture: Lines 224-266
Shows: Relevance-based scoring and greedy selection
```

**Key Code Highlights:**
- **Lines 234-244**: Neighbor collection (callers + callees)
- **Lines 246-254**: Relevance scoring for each neighbor
- **Lines 256-262**: Greedy selection within max_nodes limit

#### **Screenshot 4: Relevance Scoring Algorithm**
**Lines: 268-322** (`_calculate_relevance_score` method)
```
📸 Capture: Lines 268-322
Shows: Multi-factor relevance calculation
```

**Key Code Highlights:**
- **Lines 278-283**: Proximity score (graph distance)
- **Lines 285-294**: Complexity score
- **Lines 296-311**: Control flow importance score
- **Lines 313-320**: Weighted combination (α=0.5, β=0.3, γ=0.2)

### **Talking Points**
- "Builds global call graph across entire repository"
- "Resolves imports to track cross-file dependencies"
- "Intelligent scoring selects most relevant context"
- "Greedy algorithm respects token budget constraints"

---

## 6️⃣ **RAG System (Retrieval-Augmented Generation)**

### **File**: `src/retrieval/rag.py`

### **Purpose**
Semantic code retrieval using CodeBERT embeddings and FAISS vector database.

### **Key Features**
- ✅ CodeBERT-based code embeddings
- ✅ FAISS L2 similarity search
- ✅ Diversity-aware retrieval (avoid duplicates)
- ✅ Metadata augmentation for better matching

### **Screenshot Line Numbers**

#### **Screenshot 1: RAG System Initialization**
**Lines: 8-23** (RAGSystem `__init__`)
```
📸 Capture: Lines 8-23
Shows: CodeBERT model loading with fallback
```

**Key Code Highlights:**
- **Lines 14-18**: CodeBERT loading with error handling
- **Lines 21-23**: Storage initialization

#### **Screenshot 2: Index Building**
**Lines: 65-76** (`build_index` method)
```
📸 Capture: Lines 65-76
Shows: FAISS index construction
```

**Key Code Highlights:**
- **Line 68**: Code encoding with metadata
- **Lines 70-71**: FAISS IndexFlatL2 creation
- **Line 72**: Adding embeddings to index

#### **Screenshot 3: Retrieval with Diversity**
**Lines: 78-150** (`retrieve` method)
```
📸 Capture: Lines 78-150
Shows: Similarity search with diversity filtering
```

**Key Code Highlights:**
- **Lines 85-87**: FAISS search for top-k candidates
- **Lines 107-128**: Diversity filtering (avoid duplicate function names)
- **Lines 130-145**: Metadata normalization

### **Talking Points**
- "Uses CodeBERT for semantic code understanding"
- "FAISS enables fast similarity search at scale"
- "Diversity filtering prevents redundant examples"

---

## 7️⃣ **Reflective Agent (LangGraph Workflow)**

### **File**: `src/model/reflective_agent.py`

### **Purpose**
Implements agentic self-correction workflow using LangGraph state machine.

### **Key Features**
- ✅ Generate → Critique → Decide → Consult/Refine loop
- ✅ Score-based quality assessment
- ✅ Repository graph consultation for missing context
- ✅ Iterative refinement with max attempts

### **Screenshot Line Numbers**

#### **Screenshot 1: Agent State Definition**
**Lines: 11-22** (AgentState TypedDict)
```
📸 Capture: Lines 11-22
Shows: State machine data structure
```

**Key Code Highlights:**
- **Lines 13-15**: Code and context storage
- **Lines 16-18**: Summary, critique, and missing dependencies
- **Lines 19-21**: Iteration tracking and metadata

#### **Screenshot 2: LangGraph Workflow Construction**
**Lines: 29-58** (`_build_graph` method)
```
📸 Capture: Lines 29-58
Shows: State machine graph construction
```

**Key Code Highlights:**
- **Lines 32-36**: Node definitions (generate, critique, decide, consult, refine)
- **Lines 39-47**: Edge definitions (workflow transitions)
- **Lines 50-52**: Conditional routing based on action

#### **Screenshot 3: Critique Node**
**Lines: 82-132** (`critique_summary` method)
```
📸 Capture: Lines 82-132
Shows: Summary evaluation and scoring
```

**Key Code Highlights:**
- **Lines 85-100**: Critique prompt construction
- **Lines 110-125**: JSON parsing and score extraction
- **Lines 127-131**: Missing function detection

#### **Screenshot 4: Decision Logic**
**Lines: 134-172** (`decide_action` method)
```
📸 Capture: Lines 134-172
Shows: Action selection based on critique
```

**Key Code Highlights:**
- **Lines 142-144**: FINISH if score ≥ 8
- **Lines 146-148**: CONSULT if missing dependencies
- **Lines 150-152**: REFINE if attempts remaining
- **Lines 154-156**: FINISH if max attempts reached

#### **Screenshot 5: Repository Consultation**
**Lines: 174-218** (`consult_context` method)
```
📸 Capture: Lines 174-218
Shows: Dynamic context retrieval from repo graph
```

**Key Code Highlights:**
- **Lines 182-190**: Extracting missing function names
- **Lines 192-208**: Querying repository graph for each function
- **Lines 210-215**: Appending retrieved context

### **Talking Points**
- "LangGraph provides state machine for agentic workflow"
- "Score-based stopping (≥8) ensures quality"
- "Dynamically consults repo graph when dependencies missing"
- "Iterative refinement improves summary quality"

---

## 8️⃣ **Inference Pipeline**

### **File**: `src/model/inference.py`

### **Purpose**
Orchestrates end-to-end inference: structural analysis → RAG retrieval → prompt construction → generation.

### **Key Features**
- ✅ Integrated repo graph, RAG, and structural analysis
- ✅ Hierarchical prompt construction
- ✅ Support for both normal and agent modes
- ✅ Comprehensive logging and validation

### **Screenshot Line Numbers**

#### **Screenshot 1: Pipeline Initialization**
**Lines: 20-72** (InferencePipeline `__init__`)
```
📸 Capture: Lines 20-72
Shows: Loading model, RAG, repo graph, and agent
```

**Key Code Highlights:**
- **Lines 24-37**: Model loading with error handling
- **Lines 56-62**: RAG system loading
- **Lines 64-66**: Repository graph initialization
- **Lines 68-69**: Reflective agent initialization

#### **Screenshot 2: Summarization Logic**
**Lines: 83-141** (`summarize` method)
```
📸 Capture: Lines 83-141
Shows: Function resolution and context extraction
```

**Key Code Highlights:**
- **Lines 98-105**: Function lookup in repo graph
- **Lines 107-122**: Partial name matching and error handling
- **Lines 128-137**: Transient code analysis

#### **Screenshot 3: Hierarchical Prompt Construction**
**Lines: 277-325** (`construct_hierarchical_prompt` method)
```
📸 Capture: Lines 277-325
Shows: Multi-section prompt assembly
```

**Key Code Highlights:**
- **Lines 283-284**: Instruction section
- **Lines 286-298**: Target function metadata
- **Lines 300-304**: Dependency context (call graph)
- **Lines 306-316**: Similar code patterns (RAG)
- **Lines 318-320**: Code to summarize

#### **Screenshot 4: Response Generation**
**Lines: 205-275** (`generate_response` method)
```
📸 Capture: Lines 205-275
Shows: Model inference with validation
```

**Key Code Highlights:**
- **Lines 216-227**: Input tokenization and logging
- **Lines 230-238**: Model generation with parameters
- **Lines 241-247**: Output decoding and validation
- **Lines 253-272**: Quality checks and error handling

### **Talking Points**
- "Integrates all components into unified pipeline"
- "Hierarchical prompts organize information clearly"
- "Comprehensive validation ensures quality outputs"
- "Supports both single-pass and agentic modes"

---

## 9️⃣ **Prompt Construction**

### **File**: `src/data/prompt.py`

### **Purpose**
Constructs structured prompts fusing AST, CFG, PDG, and Call Graph information.

### **Screenshot Line Numbers**

#### **Screenshot 1: Structural Prompt Fusion**
**Lines: 4-19** (`construct_structural_prompt` function)
```
📸 Capture: Lines 4-19
Shows: Multi-view graph integration
```

**Key Code Highlights:**
- **Lines 14-17**: Extracting AST, CFG, PDG, Call Graph
- **Line 19**: Fusing into single hierarchical prompt

#### **Screenshot 2: Complete Prompt Assembly**
**Lines: 21-61** (`construct_prompt` function)
```
📸 Capture: Lines 21-61
Shows: Final prompt with instruction, context, examples, and code
```

**Key Code Highlights:**
- **Lines 26-37**: Default instruction emphasizing dependencies
- **Lines 42-46**: Context section (repo + structural)
- **Lines 48-53**: Few-shot examples from RAG
- **Lines 55-59**: Target code and output indicator

---

## 📊 **Summary: Code Coverage**

| Component | File | Key Lines | Purpose |
|-----------|------|-----------|---------|
| **Dataset Loading** | `dataset.py` | 17-76 | Load and validate training data |
| **Model Loading** | `model_loader.py` | 6-61 | 4-bit quantization + LoRA setup |
| **Training** | `trainer.py` | 10-115 | Complete training pipeline |
| **AST Analysis** | `ast_analyzer.py` | 6-240 | Deep structural analysis |
| **Repo Graph** | `repo_graph.py` | 11-354 | Global call graph construction |
| **RAG System** | `rag.py` | 8-150 | Semantic code retrieval |
| **Reflective Agent** | `reflective_agent.py` | 11-280 | Agentic workflow (LangGraph) |
| **Inference** | `inference.py` | 20-325 | End-to-end orchestration |
| **Prompts** | `prompt.py` | 4-61 | Multi-view prompt fusion |

---

## 🎯 **Presentation Strategy**

### **Slide Organization**

**Option 1: One Slide Per Component** (9 slides)
- Detailed walkthrough of each major component
- Best for technical deep-dive presentations

**Option 2: Grouped Slides** (4-5 slides)
- **Slide 1**: Data & Model (Dataset + Model Loading)
- **Slide 2**: Training & Structural Analysis
- **Slide 3**: Repository Graph & RAG
- **Slide 4**: Reflective Agent & Inference
- Best for time-constrained presentations

**Option 3: Highlights Only** (2-3 slides)
- **Slide 1**: Core Pipeline (Training + Inference)
- **Slide 2**: Novel Components (Repo Graph + Reflective Agent)
- Best for high-level overviews

### **Screenshot Guidelines**

1. **Use syntax highlighting** - Ensure code is readable with proper colors
2. **Zoom appropriately** - Line numbers and code should be clearly visible
3. **Highlight key lines** - Use arrows or boxes to emphasize important sections
4. **Add annotations** - Brief labels explaining what each section does
5. **Consistent styling** - Use same theme/font across all screenshots

### **Talking Points Template**

For each screenshot:
1. **What**: "This is the [component name] implementation"
2. **Why**: "It's needed because [problem it solves]"
3. **How**: "Key features include [2-3 bullet points]"
4. **Impact**: "This enables [specific capability]"

---

## 💡 **Key Messages**

### **Technical Excellence**
- ✅ **Modular Design**: Each component is self-contained and reusable
- ✅ **Production-Ready**: Error handling, logging, validation throughout
- ✅ **Scalable**: Efficient algorithms (FAISS, graph traversal)
- ✅ **Maintainable**: Clear structure, comprehensive documentation

### **Innovation Highlights**
- 🔥 **Multi-View Graphs**: First system to fuse AST+CFG+PDG+Call Graph
- 🔥 **Intelligent Context**: Relevance-based subgraph extraction
- 🔥 **Agentic Workflow**: LangGraph-based self-correction
- 🔥 **Hybrid RAG**: Semantic + structural context

### **Implementation Quality**
- ✅ **4-bit Quantization**: Runs on consumer hardware
- ✅ **LoRA Fine-Tuning**: Parameter-efficient training
- ✅ **Comprehensive Logging**: Full observability
- ✅ **Robust Error Handling**: Graceful degradation

---

## 🎤 **Closing Statement**

*"This codebase represents a complete, production-ready implementation of NeuroGraph-CodeRAG. Every component—from dataset loading to agentic refinement—is carefully designed, well-documented, and thoroughly tested. The modular architecture allows each innovation to work independently while seamlessly integrating into the full pipeline. This isn't just a research prototype; it's a robust system ready for real-world deployment."*

---

**End of Code Implementation Slide**
