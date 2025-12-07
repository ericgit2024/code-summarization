# Chapter 4: Proposed Solution

## 4.1 Introduction

This chapter presents **NeuroGraph-CodeRAG**, our comprehensive solution to the graph-augmented repository-aware code summarization problem defined in Chapter 3. We describe the complete system architecture, methodology, and algorithms that address the identified challenges. The solution integrates multiple innovative components:

1. **Multi-View Structural Analysis Engine**: Extracts and serializes AST, CFG, PDG, and Call Graph representations
2. **Repository-Wide Context System**: Builds global dependency graphs and performs intelligent subgraph extraction
3. **Retrieval-Augmented Generation (RAG)**: Provides few-shot learning context from similar code examples
4. **Reflective Agentic Workflow**: Implements iterative Generate→Critique→Consult→Refine cycles using LangGraph
5. **Prompt Engineering Framework**: Translates complex graph structures into LLM-readable textual prompts

The chapter is organized as follows: Section 4.2 presents the overall system architecture, Section 4.3 details the methodology and workflow, Section 4.4 provides algorithmic specifications with pseudocode, Section 4.5 describes implementation details, and Section 4.6 discusses future enhancements to further improve the system.

---

## 4.2 System Architecture

### 4.2.1 High-Level Architecture

**Figure 4.1: NeuroGraph-CodeRAG System Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                      PRESENTATION LAYER                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Streamlit Web Interface                        │ │
│  │  • File Upload  • Function Selection  • Visualization      │ │
│  │  • CFG Display  • Call Graph Display  • Summary Output     │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    APPLICATION LOGIC LAYER                       │
│  ┌──────────────────────┐      ┌──────────────────────────────┐ │
│  │ Inference Pipeline   │◄────►│   Reflective Agent           │ │
│  │  • Orchestration     │      │   (LangGraph Workflow)       │ │
│  │  • Prompt Building   │      │   • Generate → Critique      │ │
│  │  • Model Invocation  │      │   • Decide → Consult/Refine  │ │
│  └──────────────────────┘      └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
           ↓                ↓                    ↓
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  STRUCTURAL      │  │  RETRIEVAL       │  │  MODEL           │
│  ANALYSIS        │  │  SYSTEM          │  │  INFRASTRUCTURE  │
│                  │  │                  │  │                  │
│ • RepoGraphBuild │  │ • RAG System     │  │ • Model Loader   │
│ • ASTAnalyzer    │  │ • FAISS Index    │  │ • Gemma-2b LLM   │
│ • Graph Utils    │  │ • CodeBERT       │  │ • LoRA Adapters  │
│ • CFG/PDG Gen    │  │   Encoder        │  │                  │
└──────────────────┘  └──────────────────┘  └──────────────────┘
           ↓                                         ↓
┌──────────────────────────────────────────────────────────────┐
│                    DATA & TRAINING LAYER                      │
│  • Dataset Loader  • Trainer  • Evaluation Metrics           │
│  • Custom Dataset (386 examples)  • CodeXGlue (400K+ examples)│
└──────────────────────────────────────────────────────────────┘
```

**Architecture Principles**:
1. **Modularity**: Each component has well-defined responsibilities
2. **Separation of Concerns**: Structural analysis, retrieval, and generation are independent
3. **Extensibility**: Easy to add new graph types or analysis methods
4. **Reusability**: Repository graph built once, reused for multiple queries

---

### 4.2.2 Component Descriptions

#### **4.2.2.1 Presentation Layer**

**Streamlit Web Interface** (`src/ui/app.py`)

**Responsibilities**:
- Accept user input (code upload, function selection)
- Display visualizations (CFG, call graph)
- Present generated summaries
- Provide mode selection (Normal vs. Smart Agent)

**Key Features**:
- **File Upload**: Supports single files or repository dumps
- **Function Selection**: Dropdown to choose target function
- **Visualization**: Interactive CFG and call graph displays using Graphviz
- **Mode Toggle**: Switch between fast (normal) and thorough (smart agent) summarization
- **Real-time Feedback**: Progress indicators during graph construction and generation

**Technologies**:
- Streamlit for web framework
- Graphviz for graph visualization
- Matplotlib for additional visualizations

---

#### **4.2.2.2 Application Logic Layer**

**Inference Pipeline** (`src/model/inference.py`)

**Responsibilities**:
- Orchestrate the summarization process
- Build structural prompts from graphs and context
- Invoke LLM for generation
- Coordinate with Reflective Agent when in Smart mode

**Key Methods**:
```python
class InferencePipeline:
    def __init__(self, model_path, rag_index_path):
        # Initialize model, RAG system, graph builder
        
    def generate_from_code(self, code, context, metadata):
        # Generate summary from code with structural context
        
    def build_structural_prompt(self, code, metadata, context):
        # Construct prompt with AST, CFG, PDG, Call Graph info
        
    def generate_summary(self, prompt):
        # Invoke LLM to generate summary
```

**Reflective Agent** (`src/model/reflective_agent.py`)

**Responsibilities**:
- Implement LangGraph-based agentic workflow
- Generate initial summaries
- Critique summaries for errors and missing information
- Decide whether to consult repository graph or refine
- Iteratively improve summaries

**LangGraph State Machine**:
```python
class AgentState(TypedDict):
    function_name: str
    code: str
    context: str
    summary: str
    critique: str
    missing_deps: List[str]
    consulted_functions: List[str]
    attempts: int
    max_attempts: int
    action: str  # "CONSULT", "REFINE", or "FINISH"
```

**Workflow Nodes**:
1. **generate_summary**: Create initial summary
2. **critique_summary**: Analyze summary for errors
3. **decide_action**: Determine next action based on critique
4. **consult_context**: Query repository graph for missing functions
5. **refine_summary**: Improve summary with new context

---

#### **4.2.2.3 Structural Analysis Layer**

**Repository Graph Builder** (`src/structure/repo_graph.py`)

**Responsibilities**:
- Parse entire repository to build global call graph
- Resolve cross-file dependencies and imports
- Extract relevant subgraphs for target functions
- Provide context retrieval interface

**Key Data Structures**:
```python
class RepoGraphBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()  # NetworkX directed graph
        self.function_map = {}     # Map function names to metadata
        
    def build_from_directory(self, repo_path):
        # Parse all Python files, build call graph
        
    def get_function_context(self, function_name, max_depth=2):
        # Extract subgraph: callers + callees
        
    def score_relevance(self, neighbor, target):
        # Compute relevance score for intelligent extraction
```

**AST Analyzer** (`src/structure/ast_analyzer.py`)

**Responsibilities**:
- Parse Python code into Abstract Syntax Trees
- Extract metadata (complexity, variables, parameters)
- Identify function calls and dependencies

**Extracted Metadata**:
- Cyclomatic complexity
- Number of parameters
- Return type (if annotated)
- Local variables
- Function calls (direct)
- Imported modules

**Graph Utilities** (`src/structure/graph_utils.py`)

**Responsibilities**:
- Construct Control Flow Graphs (CFG)
- Generate Program Dependence Graphs (PDG)
- Visualize graphs using Graphviz
- Serialize graphs to textual format

**Graph Construction**:
- **CFG**: Uses `py2cfg` library or custom implementation
- **PDG**: Combines control dependencies and data flow analysis
- **Visualization**: Exports to DOT format for Graphviz rendering

---

#### **4.2.2.4 Retrieval System**

**RAG System** (`src/retrieval/rag.py`)

**Responsibilities**:
- Encode code snippets into dense vectors
- Build and maintain FAISS index
- Retrieve similar code examples for few-shot learning
- Augment prompts with retrieved examples

**Architecture**:
```python
class RAGSystem:
    def __init__(self, index_path):
        self.encoder = SentenceTransformer('microsoft/codebert-base')
        self.index = faiss.read_index(index_path)
        self.examples = []  # Stored code-summary pairs
        
    def encode_code(self, code):
        # Convert code to dense vector
        
    def retrieve_similar(self, code, k=3):
        # Find top-k similar examples from index
        
    def augment_prompt(self, prompt, examples):
        # Add retrieved examples to prompt
```

**FAISS Index**:
- **Index Type**: Flat L2 (exact search) or IVF (approximate for scale)
- **Dimension**: 768 (CodeBERT embedding size)
- **Storage**: Serialized to disk for persistence

---

#### **4.2.2.5 Model Infrastructure**

**Model Loader** (`src/model/inference.py`)

**Responsibilities**:
- Load Gemma-2b base model
- Apply LoRA adapters (if fine-tuned)
- Configure generation parameters
- Manage GPU/CPU execution

**Configuration**:
```python
model_config = {
    "model_name": "google/gemma-2b",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "max_length": 4096,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True
}
```

**LoRA Fine-Tuning**:
- **Rank**: 8 or 16
- **Alpha**: 32
- **Target Modules**: Query and Value projection layers
- **Dropout**: 0.1

---

#### **4.2.2.6 Data & Training Layer**

**Dataset Loader** (`src/data/dataset.py`)

**Responsibilities**:
- Load and preprocess training data
- Handle both custom and CodeXGlue datasets
- Create train/validation/test splits
- Batch data for training

**Dataset Formats**:
- **Custom Dataset**: 386 hand-crafted examples with dependency-rich summaries
- **CodeXGlue**: 400K+ examples from CodeSearchNet (Python subset)

**Trainer** (`src/model/trainer.py`)

**Responsibilities**:
- Fine-tune Gemma-2b with LoRA
- Implement training loop with structural prompts
- Compute and log metrics
- Save checkpoints

**Training Configuration**:
- **Optimizer**: AdamW
- **Learning Rate**: 2e-4 with cosine schedule
- **Batch Size**: 4 (with gradient accumulation)
- **Epochs**: 3-5
- **Loss**: Cross-entropy for next-token prediction

---

### 4.2.3 Data Flow

**End-to-End Summarization Flow**:

```
1. User Input
   ↓
2. Repository Graph Construction
   • Parse all files in repository
   • Build global call graph (NetworkX)
   • Store function metadata
   ↓
3. Target Function Selection
   • User selects function to summarize
   • Extract function code and metadata
   ↓
4. Structural Analysis
   • Generate AST (ast module)
   • Construct CFG (py2cfg or custom)
   • Build PDG (data flow + control dependencies)
   • Extract relevant call graph subgraph
   ↓
5. Context Retrieval
   • RAG: Retrieve similar code examples (FAISS)
   • Repository: Get caller/callee information
   ↓
6. Prompt Construction
   • Serialize graphs to text
   • Add metadata (complexity, parameters)
   • Include RAG examples
   • Add repository context (callers/callees)
   ↓
7. Generation (Normal Mode)
   • Send prompt to Gemma-2b
   • Generate summary
   • Return result
   ↓
8. Generation (Smart Agent Mode)
   • Initialize AgentState
   • GENERATE: Create initial summary
   • CRITIQUE: Analyze for errors/gaps
   • DECIDE: Choose CONSULT, REFINE, or FINISH
   • CONSULT (if needed): Query repo graph for missing functions
   • REFINE: Improve summary with new context
   • Repeat until quality threshold or max iterations
   ↓
9. Output
   • Display summary to user
   • Show visualizations (CFG, call graph)
```

---

## 4.3 Methodology

### 4.3.1 Phase 1: Core Infrastructure (Completed)

#### **Step 1: Multi-View Structural Analysis**

**Objective**: Extract four complementary graph representations from code.

**Process**:
1. **AST Extraction**:
   - Use Python's `ast` module to parse code
   - Extract nodes: FunctionDef, ClassDef, If, For, While, etc.
   - Build tree structure with parent-child relationships

2. **CFG Construction**:
   - Identify basic blocks (sequences with single entry/exit)
   - Create edges for control flow (sequential, conditional, loop)
   - Label edges with conditions (for branches)
   - Handle exception handling (try/except blocks)

3. **PDG Generation**:
   - **Data Dependencies**: Track variable definitions and uses
   - **Control Dependencies**: Identify which statements control others' execution
   - Combine into unified graph

4. **Call Graph Building**:
   - Parse all files in repository
   - Identify function calls (direct, method calls, imported functions)
   - Resolve imports to link cross-file dependencies
   - Build directed graph: nodes = functions, edges = calls

**Output**: Four graph structures ($G_{\text{AST}}, G_{\text{CFG}}, G_{\text{PDG}}, G_{\text{CG}}$) for each function.

---

#### **Step 2: Prompt Engineering**

**Objective**: Translate graph structures into LLM-readable textual prompts.

**Serialization Strategy**:

**1. Metadata Section**:
```
Function: calculate_total_price
Complexity: 5
Parameters: items (List[Item]), discount (float)
Returns: float
Local Variables: subtotal, tax, final_price
```

**2. Control Flow Section**:
```
Control Flow:
- Entry → Check if items is empty
  - If empty → Return 0.0
  - If not empty → Calculate subtotal
- Calculate subtotal → Apply discount
- Apply discount → Calculate tax
- Calculate tax → Return final_price
```

**3. Data Dependencies Section**:
```
Data Dependencies:
- subtotal depends on: items
- final_price depends on: subtotal, discount, tax
- tax depends on: subtotal
```

**4. Repository Context Section**:
```
Called by:
- process_order (src/orders/processor.py): Uses this to compute order total
- generate_invoice (src/billing/invoice.py): Calls for invoice amount

Calls:
- apply_discount_code (src/pricing/discounts.py): Validates and applies discount
- calculate_tax (src/pricing/tax.py): Computes tax based on location
```

**5. Similar Examples (RAG)**:
```
Similar Example 1:
Code: [retrieved code snippet]
Summary: [existing summary]

Similar Example 2:
...
```

**6. Source Code**:
```python
def calculate_total_price(items: List[Item], discount: float) -> float:
    if not items:
        return 0.0
    subtotal = sum(item.price * item.quantity for item in items)
    discounted = apply_discount_code(subtotal, discount)
    tax = calculate_tax(discounted)
    final_price = discounted + tax
    return final_price
```

**Prompt Template**:
```
You are an expert code documentation assistant. Generate a concise, technical summary of the following Python function. The summary should:
1. Explain what the function does
2. Describe its control flow and logic
3. Mention which functions call it ("Called by")
4. Mention which functions it calls ("Calls")
5. Be 2-4 sentences, technical and precise

[Metadata Section]
[Control Flow Section]
[Data Dependencies Section]
[Repository Context Section]
[Similar Examples Section]
[Source Code Section]

Summary:
```

---

#### **Step 3: RAG System Implementation**

**Objective**: Provide few-shot learning context from similar code examples.

**Process**:
1. **Index Building**:
   - Load training dataset (code-summary pairs)
   - Encode each code snippet using CodeBERT
   - Build FAISS index from embeddings
   - Store code-summary pairs for retrieval

2. **Retrieval**:
   - Encode query code using CodeBERT
   - Search FAISS index for top-k nearest neighbors (k=3)
   - Retrieve corresponding code-summary pairs

3. **Prompt Augmentation**:
   - Add retrieved examples to prompt as few-shot demonstrations
   - Format: "Similar Example 1: [code] → [summary]"

**Benefits**:
- Guides model on desired output format
- Provides domain-specific examples
- Improves consistency across summaries

---

#### **Step 4: Model Fine-Tuning**

**Objective**: Adapt Gemma-2b to generate dependency-rich summaries.

**Process**:
1. **Data Preparation**:
   - Combine custom dataset (386 examples) with CodeXGlue (400K+ examples)
   - For each example, construct structural prompt
   - Create input-output pairs: (prompt, summary)

2. **LoRA Configuration**:
   - Apply LoRA to Query and Value projection layers
   - Rank: 8, Alpha: 32, Dropout: 0.1
   - Freeze base model, train only LoRA parameters

3. **Training**:
   - Optimizer: AdamW with learning rate 2e-4
   - Batch size: 4 with gradient accumulation (effective batch size: 16)
   - Epochs: 3-5
   - Loss: Cross-entropy for next-token prediction

4. **Validation**:
   - Monitor BLEU, ROUGE scores on validation set
   - Early stopping if validation loss plateaus
   - Save best checkpoint

**Output**: Fine-tuned model with LoRA adapters optimized for structural code summarization.

---

### 4.3.2 Phase 2: Agentic Workflow (Completed)

#### **Step 5: Reflective Agent Implementation**

**Objective**: Implement iterative Generate→Critique→Consult→Refine workflow using LangGraph.

**LangGraph State Machine**:

```
       START
         ↓
    [GENERATE]
         ↓
    [CRITIQUE]
         ↓
     [DECIDE]
      /  |  \
CONSULT REFINE FINISH
     \   |   /
      \  |  /
    [CRITIQUE]
       (loop)
```

**Node Implementations**:

**1. GENERATE Node**:
```python
def generate_summary(state: AgentState) -> AgentState:
    # Build prompt from code + context
    prompt = build_prompt(state['code'], state['context'], state['metadata'])
    
    # Generate initial summary
    summary = llm.generate(prompt)
    
    # Update state
    state['summary'] = summary
    state['attempts'] += 1
    
    return state
```

**2. CRITIQUE Node**:
```python
def critique_summary(state: AgentState) -> AgentState:
    # Critique prompt
    critique_prompt = f"""
    You are a code review expert. Analyze this summary against the source code.
    
    Code:
    {state['code']}
    
    Summary:
    {state['summary']}
    
    Check for:
    1. Factual accuracy (does summary match code?)
    2. Missing dependencies (are called functions mentioned?)
    3. Control flow correctness (is logic accurately described?)
    4. Completeness (are all important aspects covered?)
    
    Provide critique in JSON format:
    {{
        "score": 0-10,
        "issues": ["issue1", "issue2", ...],
        "missing_functions": ["func1", "func2", ...]
    }}
    """
    
    # Get critique
    critique_response = llm.generate(critique_prompt)
    critique = parse_json(critique_response)
    
    # Update state
    state['critique'] = critique
    state['missing_deps'] = critique.get('missing_functions', [])
    
    return state
```

**3. DECIDE Node**:
```python
def decide_action(state: AgentState) -> AgentState:
    critique = state['critique']
    
    # Decision logic
    if critique['score'] >= 8:
        state['action'] = 'FINISH'
    elif state['missing_deps'] and state['attempts'] < state['max_attempts']:
        state['action'] = 'CONSULT'
    elif state['attempts'] < state['max_attempts']:
        state['action'] = 'REFINE'
    else:
        state['action'] = 'FINISH'  # Max attempts reached
    
    return state
```

**4. CONSULT Node**:
```python
def consult_context(state: AgentState) -> AgentState:
    # Query repository graph for missing functions
    missing_funcs = state['missing_deps']
    additional_context = []
    
    for func_name in missing_funcs:
        if func_name not in state['consulted_functions']:
            # Retrieve function from repo graph
            func_info = repo_graph.get_function_info(func_name)
            if func_info:
                additional_context.append(f"{func_name}: {func_info['docstring']}")
                state['consulted_functions'].append(func_name)
    
    # Update context
    if additional_context:
        state['context'] += "\n\nAdditional Context:\n" + "\n".join(additional_context)
    
    return state
```

**5. REFINE Node**:
```python
def refine_summary(state: AgentState) -> AgentState:
    # Refinement prompt
    refine_prompt = f"""
    Improve the following summary based on the critique.
    
    Original Summary:
    {state['summary']}
    
    Critique:
    {state['critique']}
    
    Code:
    {state['code']}
    
    Context:
    {state['context']}
    
    Generate an improved summary addressing the issues:
    """
    
    # Generate refined summary
    refined_summary = llm.generate(refine_prompt)
    
    # Update state
    state['summary'] = refined_summary
    state['attempts'] += 1
    
    return state
```

**Workflow Execution**:
```python
def run_agent(function_name, code, context, metadata, max_attempts=5):
    # Initialize state
    state = AgentState(
        function_name=function_name,
        code=code,
        context=context,
        summary="",
        critique="",
        missing_deps=[],
        consulted_functions=[],
        attempts=0,
        max_attempts=max_attempts,
        metadata=metadata,
        action=""
    )
    
    # Run LangGraph workflow
    final_state = workflow.invoke(state)
    
    return final_state['summary']
```

---

### 4.3.3 Intelligent Subgraph Extraction

**Objective**: Select most relevant context functions within token budget.

**Algorithm**:

**1. Relevance Scoring**:
```python
def score_relevance(neighbor, target, graph):
    # Proximity score (inverse of shortest path distance)
    distance = nx.shortest_path_length(graph, target, neighbor)
    proximity_score = 1.0 / (distance + 1)
    
    # Complexity score (normalized cyclomatic complexity)
    complexity = neighbor.metadata.get('complexity', 1)
    complexity_score = min(complexity / 10.0, 1.0)
    
    # Control flow importance (is it called in loop/conditional?)
    cf_importance = neighbor.metadata.get('cf_importance', 0.5)
    
    # Weighted combination
    relevance = (
        0.5 * proximity_score +
        0.3 * complexity_score +
        0.2 * cf_importance
    )
    
    return relevance
```

**2. Greedy Selection**:
```python
def extract_subgraph(target, graph, max_tokens=1000):
    # Get all neighbors (callers + callees)
    neighbors = list(graph.predecessors(target)) + list(graph.successors(target))
    
    # Score each neighbor
    scored_neighbors = [
        (neighbor, score_relevance(neighbor, target, graph))
        for neighbor in neighbors
    ]
    
    # Sort by relevance (descending)
    scored_neighbors.sort(key=lambda x: x[1], reverse=True)
    
    # Greedily select until token budget exhausted
    selected = []
    current_tokens = 0
    
    for neighbor, score in scored_neighbors:
        neighbor_tokens = estimate_tokens(neighbor)
        if current_tokens + neighbor_tokens <= max_tokens:
            selected.append(neighbor)
            current_tokens += neighbor_tokens
        else:
            break
    
    return selected
```

---

## 4.4 Algorithms

### 4.4.1 Repository Graph Construction

**Algorithm 4.1: Build Repository Call Graph**

```
Input: repository_path (path to code repository)
Output: G_CG (directed call graph)

1: G_CG ← empty directed graph
2: function_map ← empty dictionary
3: 
4: // Phase 1: Parse all files and extract functions
5: for each file in repository_path:
6:     if file.extension == '.py':
7:         ast_tree ← parse_file(file)
8:         for each node in ast_tree:
9:             if node is FunctionDef or ClassDef:
10:                function_info ← extract_metadata(node)
11:                function_map[node.name] ← function_info
12:                G_CG.add_node(node.name, **function_info)
13:
14: // Phase 2: Identify function calls and build edges
15: for each file in repository_path:
16:     ast_tree ← parse_file(file)
17:     for each function_def in ast_tree:
18:         caller ← function_def.name
19:         for each node in function_def.body:
20:             if node is Call:
21:                 callee ← resolve_function_name(node)
22:                 if callee in function_map:
23:                     G_CG.add_edge(caller, callee)
24:
25: // Phase 3: Resolve imports
26: for each file in repository_path:
27:     imports ← extract_imports(file)
28:     for each import in imports:
29:         resolve_import_dependencies(import, G_CG)
30:
31: return G_CG
```

**Time Complexity**: $O(n \cdot m)$ where $n$ = number of files, $m$ = average file size  
**Space Complexity**: $O(f + e)$ where $f$ = number of functions, $e$ = number of call edges

---

### 4.4.2 Control Flow Graph Construction

**Algorithm 4.2: Build Control Flow Graph**

```
Input: function_ast (AST of target function)
Output: G_CFG (control flow graph)

1: G_CFG ← empty directed graph
2: entry_block ← create_basic_block("entry")
3: G_CFG.add_node(entry_block)
4: current_block ← entry_block
5:
6: function build_cfg(node, current_block):
7:     if node is Assignment or Expression:
8:         current_block.add_statement(node)
9:         return current_block
10:    
11:    if node is If:
12:        // Create blocks for condition, then-branch, else-branch
13:        cond_block ← create_basic_block("condition")
14:        then_block ← create_basic_block("then")
15:        else_block ← create_basic_block("else")
16:        merge_block ← create_basic_block("merge")
17:        
18:        G_CFG.add_edge(current_block, cond_block)
19:        G_CFG.add_edge(cond_block, then_block, label="True")
20:        G_CFG.add_edge(cond_block, else_block, label="False")
21:        
22:        then_exit ← build_cfg(node.body, then_block)
23:        else_exit ← build_cfg(node.orelse, else_block)
24:        
25:        G_CFG.add_edge(then_exit, merge_block)
26:        G_CFG.add_edge(else_exit, merge_block)
27:        
28:        return merge_block
29:    
30:    if node is While or For:
31:        // Create blocks for loop header, body, exit
32:        loop_header ← create_basic_block("loop_header")
33:        loop_body ← create_basic_block("loop_body")
34:        loop_exit ← create_basic_block("loop_exit")
35:        
36:        G_CFG.add_edge(current_block, loop_header)
37:        G_CFG.add_edge(loop_header, loop_body, label="True")
38:        G_CFG.add_edge(loop_header, loop_exit, label="False")
39:        
40:        body_exit ← build_cfg(node.body, loop_body)
41:        G_CFG.add_edge(body_exit, loop_header)  // Back edge
42:        
43:        return loop_exit
44:    
45:    if node is Return:
46:        exit_block ← create_basic_block("exit")
47:        G_CFG.add_edge(current_block, exit_block)
48:        return exit_block
49:
50: // Build CFG starting from entry
51: exit_block ← build_cfg(function_ast.body, entry_block)
52:
53: return G_CFG
```

**Time Complexity**: $O(n)$ where $n$ = number of AST nodes  
**Space Complexity**: $O(b)$ where $b$ = number of basic blocks

---

### 4.4.3 Agentic Workflow

**Algorithm 4.3: Reflective Agent Workflow**

```
Input: function_name, code, initial_context, metadata, max_attempts
Output: refined_summary

1: state ← initialize_state(function_name, code, initial_context, metadata)
2: state.max_attempts ← max_attempts
3: state.attempts ← 0
4:
5: // GENERATE initial summary
6: prompt ← build_structural_prompt(code, metadata, initial_context)
7: state.summary ← LLM.generate(prompt)
8: state.attempts ← 1
9:
10: while state.attempts < max_attempts:
11:     // CRITIQUE
12:     critique_prompt ← build_critique_prompt(code, state.summary)
13:     critique_response ← LLM.generate(critique_prompt)
14:     state.critique ← parse_critique(critique_response)
15:     
16:     // DECIDE
17:     if state.critique.score >= quality_threshold:
18:         state.action ← "FINISH"
19:         break
20:     elif state.critique.missing_functions is not empty:
21:         state.action ← "CONSULT"
22:     else:
23:         state.action ← "REFINE"
24:     
25:     // CONSULT (if needed)
26:     if state.action == "CONSULT":
27:         for each func in state.critique.missing_functions:
28:             if func not in state.consulted_functions:
29:                 func_info ← repo_graph.get_function_info(func)
30:                 if func_info is not null:
31:                     state.context ← state.context + func_info
32:                     state.consulted_functions.add(func)
33:     
34:     // REFINE
35:     refine_prompt ← build_refine_prompt(state.summary, state.critique, 
36:                                          code, state.context)
37:     state.summary ← LLM.generate(refine_prompt)
38:     state.attempts ← state.attempts + 1
39:
40: return state.summary
```

**Time Complexity**: $O(T \cdot L)$ where $T$ = max iterations, $L$ = LLM inference time  
**Space Complexity**: $O(C)$ where $C$ = context size

---

### 4.4.4 Intelligent Subgraph Extraction

**Algorithm 4.4: Extract Relevant Context Subgraph**

```
Input: target_function, G_CG (call graph), max_tokens
Output: relevant_context (list of function info)

1: // Get all neighbors (1-hop callers and callees)
2: callers ← G_CG.predecessors(target_function)
3: callees ← G_CG.successors(target_function)
4: neighbors ← callers ∪ callees
5:
6: // Score each neighbor
7: scored_neighbors ← []
8: for each neighbor in neighbors:
9:     // Compute proximity score
10:    distance ← shortest_path_length(G_CG, target_function, neighbor)
11:    proximity ← 1.0 / (distance + 1)
12:    
13:    // Compute complexity score
14:    complexity ← neighbor.metadata['cyclomatic_complexity']
15:    complexity_score ← min(complexity / 10.0, 1.0)
16:    
17:    // Compute control flow importance
18:    cf_importance ← neighbor.metadata.get('cf_importance', 0.5)
19:    
20:    // Weighted relevance score
21:    relevance ← α × proximity + β × complexity_score + γ × cf_importance
22:    
23:    scored_neighbors.append((neighbor, relevance))
24:
25: // Sort by relevance (descending)
26: scored_neighbors.sort(key=lambda x: x[1], reverse=True)
27:
28: // Greedy selection within token budget
29: selected_context ← []
30: current_tokens ← 0
31:
32: for each (neighbor, score) in scored_neighbors:
33:     neighbor_tokens ← estimate_token_count(neighbor)
34:     if current_tokens + neighbor_tokens ≤ max_tokens:
35:         selected_context.append(neighbor)
36:         current_tokens ← current_tokens + neighbor_tokens
37:     else:
38:         break  // Budget exhausted
39:
40: return selected_context
```

**Parameters**: $\alpha = 0.5, \beta = 0.3, \gamma = 0.2$ (tunable)  
**Time Complexity**: $O(n \log n)$ where $n$ = number of neighbors (due to sorting)  
**Space Complexity**: $O(n)$

---

## 4.5 Implementation Details

### 4.5.1 Technology Stack

**Core Technologies**:
- **Language**: Python 3.8+
- **LLM Framework**: Hugging Face Transformers
- **Model**: google/gemma-2b with LoRA adapters
- **Graph Library**: NetworkX for graph construction and analysis
- **Vector Search**: FAISS for efficient similarity search
- **Embeddings**: microsoft/codebert-base via SentenceTransformers
- **Agentic Framework**: LangGraph for state machine workflows
- **Web Framework**: Streamlit for interactive UI
- **Visualization**: Graphviz for CFG and call graph rendering
- **AST Parsing**: Python `ast` module
- **CFG Construction**: py2cfg library (with custom enhancements)

**Development Tools**:
- **Version Control**: Git
- **Package Management**: pip, requirements.txt
- **Testing**: pytest for unit tests
- **Logging**: Python logging module
- **Configuration**: YAML files for hyperparameters

---

### 4.5.2 Project Structure

```
NeuroGraph-CodeRAG/
├── src/
│   ├── structure/           # Structural analysis
│   │   ├── ast_analyzer.py      # AST parsing and metadata extraction
│   │   ├── graph_utils.py       # CFG/PDG construction
│   │   ├── repo_graph.py        # Repository call graph builder
│   │   └── ast_utils.py         # AST utility functions
│   ├── model/               # LLM and training
│   │   ├── inference.py         # Inference pipeline
│   │   ├── reflective_agent.py  # LangGraph agentic workflow
│   │   ├── trainer.py           # LoRA fine-tuning
│   │   └── prompts.py           # Prompt templates
│   ├── retrieval/           # RAG system
│   │   └── rag.py               # FAISS-based retrieval
│   ├── ui/                  # User interface
│   │   ├── app.py               # Streamlit main app
│   │   └── visualizations.py    # Graph visualization utilities
│   ├── data/                # Data processing
│   │   ├── dataset.py           # Dataset loading
│   │   └── preprocessing.py     # Data preprocessing
│   ├── evaluation/          # Metrics
│   │   └── metrics.py           # BLEU, ROUGE, METEOR, etc.
│   ├── scripts/             # Utility scripts
│   │   ├── build_rag_index.py   # Build FAISS index
│   │   ├── download_codexglue.py # Download CodeXGlue dataset
│   │   ├── preprocess_codexglue.py # Preprocess CodeXGlue
│   │   └── create_dataset_splits.py # Train/val/test splits
│   └── utils/               # General utilities
│       └── logging_utils.py     # Logging configuration
├── data/
│   ├── code_summary_dataset.jsonl  # Custom dataset (386 examples)
│   ├── codexglue_train.jsonl       # CodeXGlue training set
│   ├── codexglue_validation.jsonl  # CodeXGlue validation set
│   └── codexglue_test.jsonl        # CodeXGlue test set
├── models/
│   └── gemma-2b-lora/       # Fine-tuned model checkpoints
├── rag_index/
│   └── faiss_index.pkl      # FAISS index file
├── docs/
│   ├── README.md
│   ├── FEATURES.md
│   ├── WALKTHROUGH.md
│   └── architecture.puml
├── requirements.txt
└── run_codexglue_pipeline.py  # Automated pipeline script
```

---

### 4.5.3 Key Implementation Decisions

**1. Graph Library Choice: NetworkX**
- **Rationale**: Pure Python, easy to use, sufficient for repository-scale graphs
- **Alternative Considered**: igraph (faster but C dependency)
- **Trade-off**: Simplicity over maximum performance

**2. LLM Choice: Gemma-2b**
- **Rationale**: Open-source, runs on consumer hardware, good balance of size/performance
- **Alternative Considered**: GPT-4 (better but proprietary), Llama-7b (larger)
- **Trade-off**: Accessibility over maximum fluency

**3. Fine-Tuning Method: LoRA**
- **Rationale**: Parameter-efficient, fast training, easy to swap adapters
- **Alternative Considered**: Full fine-tuning (expensive), prompt-only (less effective)
- **Trade-off**: Efficiency over maximum adaptation

**4. Vector Search: FAISS**
- **Rationale**: Industry-standard, fast, supports large-scale indexing
- **Alternative Considered**: Annoy, ScaNN
- **Trade-off**: Maturity and performance

**5. Web Framework: Streamlit**
- **Rationale**: Rapid prototyping, Python-native, easy visualization
- **Alternative Considered**: Flask/FastAPI (more control but more code)
- **Trade-off**: Development speed over customization

---

### 4.5.4 Performance Optimizations

**1. Graph Construction Caching**
```python
class RepoGraphBuilder:
    def __init__(self, cache_dir=".cache"):
        self.cache_dir = cache_dir
        
    def build_from_directory(self, repo_path):
        cache_file = os.path.join(self.cache_dir, f"{hash(repo_path)}.pkl")
        
        # Check cache
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # Build graph
        graph = self._build_graph(repo_path)
        
        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(graph, f)
        
        return graph
```

**2. Batch Processing for RAG Index Building**
```python
def build_index_batched(examples, batch_size=32):
    embeddings = []
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i+batch_size]
        batch_embeddings = encoder.encode(batch)
        embeddings.append(batch_embeddings)
    
    all_embeddings = np.vstack(embeddings)
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(all_embeddings)
    return index
```

**3. Early Stopping in Agentic Loop**
```python
def decide_action(state):
    # Stop early if quality is good enough
    if state['critique']['score'] >= 8:
        return "FINISH"
    
    # Stop if no improvement in last 2 iterations
    if len(state['score_history']) >= 2:
        if state['score_history'][-1] <= state['score_history'][-2]:
            return "FINISH"
    
    # Continue refinement
    return "REFINE" if state['attempts'] < max_attempts else "FINISH"
```

---

## 4.6 Future Enhancements

### 4.6.1 Short-Term Improvements (3-6 months)

#### **Enhancement 1: Advanced Subgraph Extraction**

**Current State**: Greedy selection based on simple relevance scoring

**Proposed Improvement**:
- **Graph Embeddings**: Use graph neural networks to learn function embeddings
- **Semantic Similarity**: Combine structural proximity with semantic similarity
- **Dynamic Weighting**: Learn optimal α, β, γ weights through reinforcement learning

**Implementation**:
```python
class LearnedSubgraphExtractor:
    def __init__(self):
        self.gnn = GraphNeuralNetwork(hidden_dim=128)
        self.scorer = MLPScorer(input_dim=128)
        
    def extract_subgraph(self, target, graph, max_tokens):
        # Encode all functions using GNN
        embeddings = self.gnn.encode(graph)
        
        # Score neighbors using learned scorer
        neighbors = graph.neighbors(target)
        scores = self.scorer(embeddings[target], embeddings[neighbors])
        
        # Select top-k within budget
        selected = greedy_select(neighbors, scores, max_tokens)
        return selected
```

**Expected Benefit**: 10-15% improvement in context relevance

---

#### **Enhancement 2: Multi-Language Support**

**Current State**: Python only

**Proposed Improvement**:
- **Language-Agnostic Parser**: Use Tree-sitter for multi-language AST parsing
- **Unified Graph Representation**: Language-independent graph structures
- **Language-Specific Adapters**: Custom handling for language-specific features

**Implementation Phases**:
1. **Phase 1**: Add Java support (similar syntax to Python)
2. **Phase 2**: Add JavaScript/TypeScript (different paradigm)
3. **Phase 3**: Add C/C++ (lower-level constructs)

**Challenges**:
- Different languages have different control flow constructs
- Import/module systems vary significantly
- Need language-specific training data

**Expected Timeline**: 6 months for Java, 12 months for all four languages

---

#### **Enhancement 3: Improved Critique Mechanisms**

**Current State**: LLM-based critique with simple JSON parsing

**Proposed Improvement**:
- **Structured Critique**: Use formal verification techniques to check summary against code
- **Automated Testing**: Generate test cases from summary, verify against code
- **Consistency Checking**: Cross-reference summary claims with extracted graphs

**Implementation**:
```python
class StructuredCritic:
    def critique(self, summary, code, graphs):
        issues = []
        
        # Check 1: Verify mentioned functions exist in call graph
        mentioned_funcs = extract_function_mentions(summary)
        actual_funcs = graphs['call_graph'].nodes()
        for func in mentioned_funcs:
            if func not in actual_funcs:
                issues.append(f"Hallucinated function: {func}")
        
        # Check 2: Verify control flow descriptions match CFG
        control_flow_claims = extract_control_flow(summary)
        cfg_paths = graphs['cfg'].get_all_paths()
        for claim in control_flow_claims:
            if not verify_claim_against_cfg(claim, cfg_paths):
                issues.append(f"Incorrect control flow: {claim}")
        
        # Check 3: Verify data dependencies
        data_deps = extract_data_dependencies(summary)
        pdg_deps = graphs['pdg'].get_dependencies()
        for dep in data_deps:
            if not verify_dependency(dep, pdg_deps):
                issues.append(f"Incorrect data dependency: {dep}")
        
        return {
            'score': 10 - len(issues),
            'issues': issues
        }
```

**Expected Benefit**: 20-30% reduction in hallucinations

---

### 4.6.2 Medium-Term Enhancements (6-12 months)

#### **Enhancement 4: Incremental Graph Updates**

**Current State**: Full repository re-parsing on any change

**Proposed Improvement**:
- **Change Detection**: Monitor file system for changes
- **Incremental Parsing**: Only re-parse modified files
- **Graph Patching**: Update call graph edges affected by changes
- **Dependency Tracking**: Identify which summaries need regeneration

**Implementation**:
```python
class IncrementalGraphBuilder:
    def __init__(self, repo_path):
        self.graph = self.build_initial_graph(repo_path)
        self.file_hashes = self.compute_file_hashes(repo_path)
        
    def update(self, changed_files):
        for file in changed_files:
            # Remove old nodes/edges
            old_functions = self.get_functions_in_file(file)
            self.graph.remove_nodes_from(old_functions)
            
            # Re-parse file
            new_ast = parse_file(file)
            new_functions = extract_functions(new_ast)
            
            # Add new nodes/edges
            self.graph.add_nodes_from(new_functions)
            self.update_edges(new_functions)
            
            # Update hash
            self.file_hashes[file] = compute_hash(file)
        
        return self.graph
```

**Expected Benefit**: 10x faster updates for large repositories

---

#### **Enhancement 5: Interactive Summary Refinement**

**Current State**: Fully automated summarization

**Proposed Improvement**:
- **User Feedback Loop**: Allow users to mark incorrect parts of summary
- **Targeted Refinement**: Focus agent on specific issues identified by user
- **Preference Learning**: Learn user preferences for summary style/detail level

**Implementation**:
```python
class InteractiveAgent:
    def refine_with_feedback(self, summary, user_feedback):
        # Parse user feedback
        incorrect_parts = user_feedback['incorrect']
        missing_info = user_feedback['missing']
        
        # Build targeted refinement prompt
        prompt = f"""
        The user identified issues with this summary:
        
        Incorrect parts: {incorrect_parts}
        Missing information: {missing_info}
        
        Original summary: {summary}
        
        Generate an improved summary addressing these specific issues.
        """
        
        # Generate refined summary
        refined = self.llm.generate(prompt)
        return refined
```

**Expected Benefit**: Higher user satisfaction, personalized summaries

---

#### **Enhancement 6: Distributed Processing**

**Current State**: Single-machine processing

**Proposed Improvement**:
- **Parallel Graph Construction**: Distribute file parsing across workers
- **Batch Summarization**: Process multiple functions in parallel
- **Distributed RAG Index**: Shard FAISS index across multiple machines

**Implementation**:
```python
from multiprocessing import Pool

class DistributedGraphBuilder:
    def build_parallel(self, repo_path, num_workers=8):
        files = list_all_files(repo_path)
        
        # Distribute file parsing
        with Pool(num_workers) as pool:
            parsed_files = pool.map(parse_file, files)
        
        # Merge results
        graph = self.merge_graphs(parsed_files)
        return graph
```

**Expected Benefit**: 5-8x speedup on multi-core machines

---

### 4.6.3 Long-Term Research Directions (12+ months)

#### **Enhancement 7: Reinforcement Learning for Agent Policies**

**Current State**: Rule-based decision logic in agent

**Proposed Improvement**:
- **RL-Based Policy**: Learn when to CONSULT vs. REFINE vs. FINISH
- **Reward Signal**: Use summary quality metrics as rewards
- **Policy Optimization**: PPO or similar algorithm to optimize decision-making

**Research Questions**:
- What is the optimal reward function?
- How to balance exploration (trying new strategies) vs. exploitation (using known good strategies)?
- Can we learn repository-specific policies?

---

#### **Enhancement 8: Code-Summary Co-Training**

**Current State**: Unidirectional (code → summary)

**Proposed Improvement**:
- **Bidirectional Learning**: Train model to generate code from summaries AND summaries from code
- **Consistency Regularization**: Ensure round-trip consistency (code → summary → code)
- **Contrastive Learning**: Learn to distinguish correct vs. incorrect summaries

**Implementation Concept**:
```python
def co_training_loss(code, summary):
    # Forward: code → summary
    generated_summary = model.generate_summary(code)
    loss_forward = cross_entropy(generated_summary, summary)
    
    # Backward: summary → code
    generated_code = model.generate_code(summary)
    loss_backward = cross_entropy(generated_code, code)
    
    # Round-trip: code → summary → code
    roundtrip_code = model.generate_code(generated_summary)
    loss_consistency = mse(roundtrip_code, code)
    
    # Combined loss
    total_loss = loss_forward + loss_backward + λ * loss_consistency
    return total_loss
```

---

#### **Enhancement 9: Neurosymbolic Integration**

**Current State**: Purely neural (LLM-based)

**Proposed Improvement**:
- **Symbolic Reasoning**: Use formal methods to verify summary correctness
- **Hybrid Architecture**: Combine neural generation with symbolic verification
- **Constraint-Based Generation**: Generate summaries that provably satisfy constraints

**Research Questions**:
- How to formalize "correct summary" as logical constraints?
- Can we use SMT solvers to verify summaries?
- How to integrate symbolic feedback into neural training?

---

#### **Enhancement 10: Cross-Repository Learning**

**Current State**: Each repository analyzed independently

**Proposed Improvement**:
- **Transfer Learning**: Learn patterns from one repository, apply to another
- **Meta-Learning**: Learn to quickly adapt to new repositories
- **Repository Embeddings**: Encode entire repositories as vectors for similarity search

**Applications**:
- **Similar Repository Recommendation**: "This function is similar to X in repository Y"
- **Cross-Repo Dependency Detection**: Identify similar patterns across projects
- **Best Practice Mining**: Learn common patterns from high-quality codebases

---

## 4.7 Summary

This chapter has presented the complete NeuroGraph-CodeRAG system, including:

**System Architecture** (Section 4.2):
- 6-layer architecture: Presentation, Application Logic, Structural Analysis, Retrieval, Model Infrastructure, Data & Training
- Detailed component descriptions for each layer
- Clear data flow from user input to summary output

**Methodology** (Section 4.3):
- Phase 1: Core infrastructure (multi-view analysis, prompt engineering, RAG, fine-tuning)
- Phase 2: Agentic workflow (LangGraph-based Generate→Critique→Consult→Refine)
- Intelligent subgraph extraction with relevance scoring

**Algorithms** (Section 4.4):
- Repository call graph construction: $O(n \cdot m)$
- Control flow graph construction: $O(n)$
- Reflective agent workflow: $O(T \cdot L)$
- Intelligent subgraph extraction: $O(n \log n)$

**Implementation Details** (Section 4.5):
- Technology stack (Python, Transformers, NetworkX, FAISS, LangGraph, Streamlit)
- Project structure with 9 main modules
- Key implementation decisions and trade-offs
- Performance optimizations (caching, batching, early stopping)

**Future Enhancements** (Section 4.6):
- **Short-term** (3-6 months): Advanced subgraph extraction, multi-language support, improved critique
- **Medium-term** (6-12 months): Incremental updates, interactive refinement, distributed processing
- **Long-term** (12+ months): RL-based policies, code-summary co-training, neurosymbolic integration, cross-repository learning

The system addresses all 10 challenges identified in Chapter 3:
1. ✅ Multi-view graph integration through explicit textual prompts
2. ✅ Scalable repository-wide context via intelligent subgraph extraction
3. ✅ Hallucination mitigation through agentic critique and refinement
4. ✅ Explicit dependency extraction via call graph and structured prompts
5. ✅ Interpretability through human-readable prompts
6. ✅ Dependency-rich evaluation metrics
7. ✅ Custom dataset with dependency-rich summaries
8. ✅ Reproducible implementation with full documentation
9. ✅ Computational efficiency through caching and early stopping
10. ✅ Generalization through diverse training data and structural grounding

The following chapters will describe the experimental methodology (Chapter 5), present results (Chapter 6), and discuss findings (Chapter 7).

---

**End of Chapter 4**
