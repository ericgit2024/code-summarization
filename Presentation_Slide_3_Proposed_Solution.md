# Slide 3: Proposed Solution - NeuroGraph-CodeRAG

## 🎯 **Solution Overview**

### **NeuroGraph-CodeRAG: Graph-Augmented Agentic Code Summarization**

> *"A comprehensive system that fuses **Static Analysis**, **Graph Theory**, and **Generative AI** to produce structurally accurate, dependency-rich, repository-aware code summaries."*

**Core Innovation**: Unlike traditional approaches that treat code as flat text, NeuroGraph-CodeRAG constructs a **multi-layered understanding** through four complementary graph representations, combined with an **agentic self-correction workflow**.

---

## 🏗️ **System Architecture**

### **High-Level Architecture Diagram**

```
┌─────────────────────────────────────────────────────────────┐
│                   PRESENTATION LAYER                         │
│              Streamlit Web Interface                         │
│   • File Upload  • Function Selection  • Visualization      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                APPLICATION LOGIC LAYER                       │
│  ┌──────────────────┐      ┌──────────────────────────┐    │
│  │ Inference        │◄────►│   Reflective Agent       │    │
│  │ Pipeline         │      │   (LangGraph Workflow)   │    │
│  │ • Orchestration  │      │   • Generate → Critique  │    │
│  │ • Prompt Build   │      │   • Decide → Consult     │    │
│  └──────────────────┘      └──────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
         ↓              ↓              ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ STRUCTURAL   │  │ RETRIEVAL    │  │ MODEL        │
│ ANALYSIS     │  │ SYSTEM       │  │ LAYER        │
│              │  │              │  │              │
│ • AST        │  │ • RAG/FAISS  │  │ • Gemma-2b   │
│ • CFG        │  │ • CodeBERT   │  │ • LoRA       │
│ • PDG        │  │ • Similar    │  │   Adapters   │
│ • Call Graph │  │   Examples   │  │              │
└──────────────┘  └──────────────┘  └──────────────┘
```

**Architecture Principles**:
- ✅ **Modularity**: Each component has well-defined responsibilities
- ✅ **Separation of Concerns**: Analysis, retrieval, and generation are independent
- ✅ **Extensibility**: Easy to add new graph types or analysis methods
- ✅ **Reusability**: Repository graph built once, reused for multiple queries

---

## 🔧 **Solution Components**

### **Component 1: Multi-View Structural Analysis Engine**

**Purpose**: Extract and serialize four complementary graph representations

#### **Four Graph Types**

**1. Abstract Syntax Tree (AST)** 🌳
- **What it captures**: Syntactic structure of code
- **Extraction**: Python's `ast` module
- **Information**: Function definitions, class hierarchies, statements, expressions
- **Example**:
  ```
  FunctionDef: calculate_total
  ├── Parameters: items, discount
  ├── Body
  │   ├── If: items is empty
  │   ├── For: iterate items
  │   └── Return: final_price
  ```

**2. Control Flow Graph (CFG)** 🔀
- **What it captures**: Execution paths and control flow
- **Construction**: Identify basic blocks and control edges
- **Information**: Loops, conditionals, exception handling, execution order
- **Example**:
  ```
  Entry → Check items
    ├─ If empty → Return 0.0
    └─ If not empty → Calculate subtotal
       → Apply discount → Calculate tax → Return
  ```

**3. Program Dependence Graph (PDG)** 🔗
- **What it captures**: Data dependencies and control dependencies
- **Analysis**: Track variable definitions, uses, and control relationships
- **Information**: Which variables affect which computations
- **Example**:
  ```
  Data Dependencies:
  - final_price depends on: subtotal, discount, tax
  - tax depends on: subtotal
  - subtotal depends on: items
  ```

**4. Call Graph (Repository-Wide)** 📊
- **What it captures**: Inter-procedural function call relationships
- **Scope**: Entire repository (cross-file dependencies)
- **Resolution**: Import analysis and symbol resolution
- **Example**:
  ```
  calculate_total
  ├─ Called by: process_order, generate_invoice
  └─ Calls: apply_discount, calculate_tax
  ```

#### **Graph Serialization to Text**

**Challenge**: LLMs need textual input, not graph objects

**Solution**: Structured textual representation

```
=== METADATA ===
Function: calculate_total_price
Complexity: 5
Parameters: items (List[Item]), discount (float)
Returns: float

=== CONTROL FLOW ===
Entry → Check if items is empty
  - If empty → Return 0.0
  - If not empty → Calculate subtotal
→ Apply discount → Calculate tax → Return final_price

=== DATA DEPENDENCIES ===
- subtotal depends on: items
- final_price depends on: subtotal, discount, tax

=== REPOSITORY CONTEXT ===
Called by:
- process_order (src/orders/processor.py)
- generate_invoice (src/billing/invoice.py)

Calls:
- apply_discount_code (src/pricing/discounts.py)
- calculate_tax (src/pricing/tax.py)

=== SOURCE CODE ===
[actual code here]
```

---

### **Component 2: Repository-Wide Context System**

**Purpose**: Build global dependency graphs and extract relevant context

#### **Repository Graph Construction**

**Process**:
1. **Parse entire repository**: All `.py` files
2. **Build global call graph**: NetworkX directed graph
   - Nodes = Functions/Methods
   - Edges = Function calls
3. **Resolve cross-file dependencies**: Import analysis
4. **Store metadata**: Complexity, parameters, docstrings

**Data Structure**:
```python
G_CG = {
    'nodes': {
        'calculate_total': {
            'file': 'src/pricing/calculator.py',
            'complexity': 5,
            'parameters': ['items', 'discount'],
            'calls': ['apply_discount', 'calculate_tax'],
            'called_by': ['process_order', 'generate_invoice']
        },
        ...
    },
    'edges': [
        ('calculate_total', 'apply_discount'),
        ('calculate_total', 'calculate_tax'),
        ...
    ]
}
```

#### **Intelligent Subgraph Extraction**

**Problem**: Can't include entire repository in prompt (token limit: 4,096)

**Solution**: Relevance-based scoring and greedy selection

**Relevance Scoring Function**:
```
Relevance(neighbor, target) = 
    α × Proximity(neighbor, target) +      // Closer in call graph = higher
    β × Complexity(neighbor) +             // More complex = more important
    γ × ControlFlowImportance(neighbor)    // Called in loops/conditionals = higher

Where: α = 0.5, β = 0.3, γ = 0.2
```

**Algorithm**:
1. Get all neighbors (callers + callees)
2. Score each neighbor by relevance
3. Sort by score (descending)
4. Greedily select until token budget exhausted

**Example**:
```
Target: calculate_total
Neighbors: [process_order, generate_invoice, apply_discount, calculate_tax]

Scores:
- apply_discount: 0.85 (direct callee, high complexity)
- calculate_tax: 0.82 (direct callee, moderate complexity)
- process_order: 0.65 (direct caller, moderate complexity)
- generate_invoice: 0.60 (direct caller, low complexity)

Selected (within token budget): apply_discount, calculate_tax, process_order
```

---

### **Component 3: Retrieval-Augmented Generation (RAG)**

**Purpose**: Provide few-shot learning context from similar code examples

#### **RAG System Architecture**

```
Input Code
    ↓
[CodeBERT Encoder] → Dense Vector (768-dim)
    ↓
[FAISS Index Search] → Top-k Similar Examples (k=3)
    ↓
[Retrieve Code-Summary Pairs]
    ↓
[Augment Prompt with Examples]
```

#### **Implementation Details**

**1. Index Building** (Offline):
```python
# Load training dataset
dataset = load_dataset('code_summary_dataset.jsonl')

# Encode all code snippets
encoder = SentenceTransformer('microsoft/codebert-base')
embeddings = encoder.encode([ex['code'] for ex in dataset])

# Build FAISS index
index = faiss.IndexFlatL2(768)  # 768 = CodeBERT dimension
index.add(embeddings)

# Save index
faiss.write_index(index, 'rag_index.pkl')
```

**2. Retrieval** (Online):
```python
# Encode query code
query_embedding = encoder.encode([input_code])

# Search for top-k similar
distances, indices = index.search(query_embedding, k=3)

# Retrieve examples
similar_examples = [dataset[i] for i in indices[0]]
```

**3. Prompt Augmentation**:
```
Similar Example 1:
Code: def validate_email(email): ...
Summary: Validates email format using regex. Called by register_user(). 
         Calls check_domain().

Similar Example 2:
...

Now summarize this code:
[target code]
```

**Benefits**:
- ✅ Guides model on desired output format
- ✅ Provides domain-specific examples
- ✅ Improves consistency across summaries
- ✅ Helps with few-shot learning

---

### **Component 4: Reflective Agentic Workflow (LangGraph)**

**Purpose**: Iteratively generate, critique, and refine summaries to reduce hallucinations

#### **The Cognitive Cycle**

```
        START
          ↓
    [1. GENERATE]
    Create initial summary
          ↓
    [2. CRITIQUE]
    Analyze for errors/gaps
          ↓
    [3. DECIDE]
    Choose next action
       /  |  \
      /   |   \
CONSULT REFINE FINISH
   |      |      |
   |      |      └─→ DONE
   |      |
   └──────┴─→ [Back to CRITIQUE]
              (Iterative Loop)
```

#### **Node Implementations**

**1. GENERATE Node**
```
Input: Code + Context + Metadata
Process: Build structural prompt → Invoke LLM
Output: Initial summary
```

**2. CRITIQUE Node**
```
Critique Prompt:
"You are a code review expert. Check this summary against the code:

Code: [source code]
Summary: [generated summary]

Check for:
1. Factual accuracy (does summary match code?)
2. Missing dependencies (are called functions mentioned?)
3. Control flow correctness (is logic accurate?)
4. Completeness (all important aspects covered?)

Provide critique in JSON:
{
  "score": 0-10,
  "issues": ["issue1", "issue2"],
  "missing_functions": ["func1", "func2"]
}"

Output: Critique with score and identified issues
```

**3. DECIDE Node**
```
Decision Logic:
IF critique.score >= 8:
    action = FINISH (quality threshold met)
ELIF missing_functions AND attempts < max_attempts:
    action = CONSULT (need more context)
ELIF attempts < max_attempts:
    action = REFINE (improve with existing context)
ELSE:
    action = FINISH (max attempts reached)
```

**4. CONSULT Node**
```
Process:
1. Extract missing_functions from critique
2. Query repository graph for each missing function
3. Retrieve function metadata (signature, docstring, complexity)
4. Append to context
5. Return to CRITIQUE with enhanced context
```

**5. REFINE Node**
```
Refinement Prompt:
"Improve this summary based on the critique:

Original Summary: [current summary]
Critique: [identified issues]
Code: [source code]
Context: [available context]

Generate improved summary addressing the issues."

Output: Refined summary
```

#### **Example Execution**

**Iteration 1**:
- **GENERATE**: "Calculates total price with discount and tax"
- **CRITIQUE**: Score 4/10. Missing: "Which functions call this? What does apply_discount do?"
- **DECIDE**: CONSULT (missing functions identified)
- **CONSULT**: Retrieve `apply_discount` and `process_order` from repo graph

**Iteration 2**:
- **REFINE**: "Calculates total price by applying discount via apply_discount() and computing tax. Called by process_order() and generate_invoice()."
- **CRITIQUE**: Score 8/10. Good coverage, minor wording issues.
- **DECIDE**: FINISH (score >= 8)

**Final Output**: Refined, dependency-rich summary

---

### **Component 5: Prompt Engineering Framework**

**Purpose**: Translate complex graph structures into LLM-readable prompts

#### **Structured Prompt Template**

```
You are an expert code documentation assistant. Generate a concise, 
technical summary of the following Python function. The summary should:
1. Explain what the function does
2. Describe its control flow and logic
3. Mention which functions call it ("Called by")
4. Mention which functions it calls ("Calls")
5. Be 2-4 sentences, technical and precise

=== METADATA ===
[Function metadata: name, complexity, parameters, returns]

=== CONTROL FLOW ===
[Serialized CFG: execution paths, branches, loops]

=== DATA DEPENDENCIES ===
[Serialized PDG: variable dependencies]

=== REPOSITORY CONTEXT ===
[Call graph context: callers and callees with descriptions]

=== SIMILAR EXAMPLES ===
[RAG-retrieved examples: code → summary pairs]

=== SOURCE CODE ===
[Actual source code]

Summary:
```

#### **Prompt Design Principles**

1. **Hierarchical Structure**: Organize information in logical sections
2. **Explicit Instructions**: Clear requirements for output format
3. **Rich Context**: Multiple views of the same code
4. **Few-Shot Learning**: Include similar examples
5. **Constraint Specification**: Length, tone, required elements

---

## 🔄 **End-to-End Workflow**

### **Complete Summarization Pipeline**

```
1. USER INPUT
   ↓
   Upload repository or code file
   Select target function

2. REPOSITORY ANALYSIS
   ↓
   Parse all .py files
   Build global call graph (NetworkX)
   Store function metadata

3. STRUCTURAL ANALYSIS
   ↓
   Extract AST (Python ast module)
   Construct CFG (control flow paths)
   Generate PDG (data dependencies)
   Extract call graph subgraph

4. CONTEXT RETRIEVAL
   ↓
   RAG: Retrieve similar examples (FAISS + CodeBERT)
   Repository: Get callers/callees (intelligent extraction)

5. PROMPT CONSTRUCTION
   ↓
   Serialize graphs to text
   Add metadata (complexity, parameters)
   Include RAG examples
   Add repository context

6. GENERATION MODE SELECTION
   ↓
   ┌─────────────────┬──────────────────────┐
   │  NORMAL MODE    │   SMART AGENT MODE   │
   ├─────────────────┼──────────────────────┤
   │ Single LLM call │ Iterative workflow   │
   │ Fast (~2 sec)   │ Thorough (~8-10 sec) │
   │ Good quality    │ Higher quality       │
   └─────────────────┴──────────────────────┘

7. OUTPUT
   ↓
   Display summary
   Show visualizations (CFG, call graph)
   Provide metadata
```

---

## 🎯 **How Our Solution Addresses Each Problem**

### **Mapping Solutions to Problems**

| **Problem (from Slide 2)** | **Our Solution** |
|----------------------------|------------------|
| **Issue 1: Multi-View Graph Integration** | ✅ Serialize 4 graph types into structured text sections; hierarchical prompt organization |
| **Issue 2: Scalable Context Extraction** | ✅ Relevance-based scoring; greedy selection within token budget; one-time graph construction |
| **Issue 3: Hallucination Mitigation** | ✅ Agentic critique-and-refine workflow; explicit verification against source code |
| **Issue 4: Explicit Dependency Extraction** | ✅ Call graph with import resolution; dedicated "Repository Context" prompt section |
| **Issue 5: Interpretability vs. Performance** | ✅ Explicit structural prompts (interpretable) + LoRA fine-tuning (performance) |

---

## 🔬 **Technical Innovations**

### **Novel Contributions**

**1. Multi-View Prompt Fusion**
- **Innovation**: First system to explicitly serialize AST + CFG + PDG + Call Graph into textual prompts
- **Advantage**: Interpretable (can trace summary claims to structural elements)
- **Contrast**: GNN methods use opaque embeddings

**2. Repository-Aware Context**
- **Innovation**: Global call graph with intelligent subgraph extraction
- **Advantage**: Summaries include cross-file dependencies
- **Contrast**: Existing methods analyze functions in isolation

**3. Agentic Self-Correction**
- **Innovation**: LangGraph-based Generate→Critique→Consult→Refine workflow
- **Advantage**: Reduces hallucinations through iterative verification
- **Contrast**: Traditional methods use single-pass generation

**4. Hybrid RAG + Structural Prompting**
- **Innovation**: Combine semantic retrieval (RAG) with structural analysis (graphs)
- **Advantage**: Few-shot learning + deep structural understanding
- **Contrast**: RAG systems typically use only semantic similarity

**5. Relevance-Based Context Selection**
- **Innovation**: Multi-factor scoring (proximity + complexity + control flow importance)
- **Advantage**: Maximizes information density within token budget
- **Contrast**: Naive approaches include all neighbors or use simple heuristics

---

## 🛠️ **Implementation Stack**

### **Technologies Used**

**Core Framework**:
- **Language**: Python 3.8+
- **LLM**: Gemma-2b (Google, 2B parameters)
- **Fine-Tuning**: LoRA (Low-Rank Adaptation)

**Structural Analysis**:
- **AST**: Python `ast` module
- **CFG**: Custom implementation / `py2cfg`
- **PDG**: Data flow analysis + control dependencies
- **Call Graph**: NetworkX + custom import resolver

**Retrieval System**:
- **Encoder**: CodeBERT (`microsoft/codebert-base`)
- **Vector DB**: FAISS (Facebook AI Similarity Search)
- **Embedding Dimension**: 768

**Agentic Workflow**:
- **Framework**: LangGraph (state machine for LLM workflows)
- **State Management**: TypedDict for agent state
- **Workflow**: Directed graph with conditional edges

**UI & Visualization**:
- **Web Framework**: Streamlit
- **Graph Visualization**: Graphviz (DOT format)
- **Plotting**: Matplotlib

**Training**:
- **Framework**: HuggingFace Transformers + PEFT
- **Optimization**: AdamW with cosine schedule
- **Quantization**: 4-bit (for memory efficiency)

---

## 📊 **System Capabilities**

### **What the System Can Do**

✅ **Analyze entire repositories** (up to 1,000 files optimized)
✅ **Extract 4 graph types** (AST, CFG, PDG, Call Graph)
✅ **Resolve cross-file dependencies** (import analysis)
✅ **Generate dependency-rich summaries** ("Called by", "Calls")
✅ **Visualize control flow** (interactive CFG display)
✅ **Self-correct hallucinations** (agentic critique workflow)
✅ **Provide interpretable prompts** (can trace summary to structural elements)
✅ **Support two modes** (Fast Normal / Thorough Smart Agent)

### **Key Metrics**

- **Context Window**: 4,096 tokens (Gemma-2b)
- **Repository Size**: Optimized for ≤ 1,000 files
- **Inference Time**: 
  - Normal Mode: ~2 seconds
  - Smart Agent Mode: ~8-10 seconds
- **Model Size**: 2B parameters (accessible on consumer hardware)
- **Training Data**: 386 custom + 400K+ CodeXGlue examples
- **RAG Retrieval**: Top-3 similar examples
- **Max Agent Iterations**: 5 (configurable)

---

## 🎯 **Advantages Over Existing Approaches**

### **Comparison with State-of-the-Art**

| **Aspect** | **GraphCodeBERT** | **HA-ConvGNN** | **CodeT5** | **NeuroGraph-CodeRAG** |
|------------|-------------------|----------------|------------|------------------------|
| **Structural Info** | Data flow only | AST + Call Graph | None | AST + CFG + PDG + CG |
| **Integration** | Implicit (pre-training) | GNN embeddings | Token sequence | Explicit prompts |
| **Repository Context** | ❌ No | ✅ Yes (class-level) | ❌ No | ✅ Yes (repo-wide) |
| **Dependency Info** | ❌ No | ❌ No | ❌ No | ✅ Yes (explicit) |
| **Hallucination Control** | ❌ No | ❌ No | ❌ No | ✅ Yes (agentic critique) |
| **Interpretability** | ❌ Low | ❌ Low | ❌ Low | ✅ High (explicit prompts) |
| **Fine-Tuning** | Full model | Full model | Full model | LoRA (efficient) |

---

## 🔮 **Future Enhancements**

### **Planned Improvements**

**1. Multi-Language Support**
- Extend to Java, C++, JavaScript
- Language-specific parsers and analysis tools

**2. Dynamic Analysis Integration**
- Runtime behavior capture
- Execution tracing for complex logic

**3. Incremental Graph Updates**
- Efficient re-computation when code changes
- Caching and differential analysis

**4. Advanced Metrics**
- Dependency coverage metric
- Structural accuracy validation
- Automated factual consistency checking

**5. Production Features**
- REST API endpoints
- IDE integration (VSCode, PyCharm)
- Continuous documentation generation

---

## 💡 **Key Takeaways**

### **What Makes NeuroGraph-CodeRAG Unique**

1. **🔀 Multi-View Understanding**: Four complementary graph types (not just one)
2. **🌐 Repository-Wide Scope**: Global context (not function-level isolation)
3. **🤖 Agentic Self-Correction**: Iterative refinement (not single-pass generation)
4. **🔍 Explicit Dependencies**: "Called by" and "Calls" (not implicit)
5. **📖 Interpretable**: Traceable prompts (not black-box embeddings)
6. **⚡ Practical**: Runs on consumer hardware (not requiring massive compute)

**Bottom Line**: We're not just improving existing methods—we're fundamentally rethinking how to combine program analysis with generative AI.

---

## 🎤 **Transition to Next Slide**

"Now that you understand our proposed solution and its architecture, let's dive into the implementation details and see how we built this system in practice..."

---

## 📝 **Speaker Notes**

### **Opening (30 seconds)**
- Start with the solution overview quote
- Emphasize **"multi-layered understanding"** as the key differentiator
- Use the architecture diagram to show the big picture

### **Component Walkthrough (4 minutes)**
- Spend ~45 seconds on each of the 5 components
- For each component:
  1. State its purpose clearly
  2. Show a concrete example
  3. Explain why it's necessary
- **Most important**: Component 4 (Reflective Agent)—this is your novel contribution

### **Workflow Demonstration (1.5 minutes)**
- Walk through the end-to-end pipeline step by step
- Use the iteration example for the Reflective Agent
- Show how all components work together

### **Problem-Solution Mapping (1 minute)**
- Use the comparison table
- Explicitly connect each solution back to problems from Slide 2
- This shows you've addressed every challenge you identified

### **Key Messages to Emphasize**

1. **"Four complementary graph types"** (not just one view)
2. **"Repository-wide context"** (not function-level)
3. **"Agentic self-correction"** (not single-pass)
4. **"Explicit and interpretable"** (not black-box)
5. **"Practical and accessible"** (runs on consumer hardware)

### **Anticipated Questions**

**Q: Why Gemma-2b instead of GPT-4?**
- A: Accessibility (open-source, runs locally), fine-tunable, sufficient capability with structural prompts

**Q: How long does repository graph construction take?**
- A: One-time cost: ~10-30 seconds for 1,000 files; reused for all queries

**Q: What if the critique is wrong?**
- A: Max iterations limit prevents infinite loops; final summary still better than single-pass

**Q: How do you handle dynamic calls (getattr, callbacks)?**
- A: Static analysis limitation; focus on statically resolvable calls; dynamic calls noted as limitation

**Q: Why LangGraph instead of custom loop?**
- A: Provides state management, conditional routing, visualization, and extensibility

### **Visual Aids to Use**

1. **Architecture Diagram**: Show at beginning, refer back throughout
2. **Graph Examples**: Visual representations of AST, CFG, PDG, Call Graph
3. **Agentic Workflow Diagram**: The circular Generate→Critique→Decide→Consult/Refine flow
4. **Iteration Example**: Before/after summaries showing improvement
5. **Comparison Table**: NeuroGraph-CodeRAG vs. existing approaches

### **Timing Breakdown**

- **Introduction**: 30 seconds
- **Component 1 (Structural Analysis)**: 1 minute
- **Component 2 (Repository Context)**: 45 seconds
- **Component 3 (RAG)**: 45 seconds
- **Component 4 (Reflective Agent)**: 1.5 minutes ⭐ (most important)
- **Component 5 (Prompt Engineering)**: 30 seconds
- **End-to-End Workflow**: 1 minute
- **Problem-Solution Mapping**: 1 minute
- **Advantages & Takeaways**: 1 minute

**Total**: ~8 minutes (adjust based on your time allocation)

### **Engagement Strategies**

1. **Ask rhetorical questions**: "How do we solve the hallucination problem?"
2. **Use concrete examples**: The iteration example is very effective
3. **Build on previous slides**: "Remember Issue 3 from Slide 2? Here's how we solve it..."
4. **Show enthusiasm**: This is YOUR innovation—be excited about it!

### **Common Pitfalls to Avoid**

- ❌ Don't get lost in implementation details (save for next slide)
- ❌ Don't assume audience knows LangGraph (explain briefly)
- ❌ Don't skip the problem-solution mapping (critical for coherence)
- ✅ **Do** use the iteration example (makes agentic workflow concrete)
- ✅ **Do** emphasize interpretability (key differentiator)
- ✅ **Do** connect back to Slide 2 problems frequently

---

## 🎨 **Visual Design Recommendations**

### **Must-Have Visuals**

**1. System Architecture Diagram**
- Use the provided ASCII diagram or create a cleaner version
- Color-code layers (Presentation=Blue, Logic=Green, Infrastructure=Orange)
- Show data flow with arrows

**2. Four Graph Types Visualization**
```
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│   AST    │  │   CFG    │  │   PDG    │  │   CG     │
│  (Tree)  │  │ (Flow)   │  │  (Deps)  │  │ (Calls)  │
└──────────┘  └──────────┘  └──────────┘  └──────────┘
```
Use icons or small diagrams for each

**3. Agentic Workflow Cycle**
```
     GENERATE
         ↓
     CRITIQUE
         ↓
      DECIDE
     /   |   \
CONSULT REFINE FINISH
     \   |   /
      (loop)
```
Make this circular/cyclical to emphasize iteration

**4. Before/After Example**
```
❌ Before (Normal LLM):
"Calculates total price"

✅ After (NeuroGraph-CodeRAG):
"Calculates total price by iterating through items, 
applying discount via apply_discount_code(), and 
computing tax via calculate_tax(). Called by 
process_order() and generate_invoice()."
```

**5. Comparison Table**
- Use checkmarks (✅) and crosses (❌) for visual impact
- Highlight your column in green

### **Color Scheme**

- 🔵 **Blue**: Architecture/System components
- 🟢 **Green**: Solutions/Advantages
- 🟡 **Yellow**: Processes/Workflows
- 🟠 **Orange**: Examples/Demonstrations
- 🔴 **Red**: Contrasts with existing approaches (sparingly)

### **Animation Suggestions** (if using PowerPoint/Keynote)

1. **Architecture Diagram**: Build layer by layer (bottom-up)
2. **Agentic Workflow**: Animate the cycle to show iteration
3. **Comparison Table**: Reveal row by row
4. **Before/After**: Show "Before" first, then reveal "After"

---

## 📚 **Technical Terms to Define**

Make sure to briefly explain:
- **LangGraph**: State machine framework for LLM workflows
- **LoRA**: Low-Rank Adaptation (efficient fine-tuning)
- **FAISS**: Vector similarity search library
- **CodeBERT**: Pre-trained model for code understanding
- **NetworkX**: Python library for graph analysis
- **Greedy Selection**: Algorithm that makes locally optimal choices

*Consider having a glossary backup slide*

---

## 🔗 **Connection to Other Slides**

**From Slide 2 (Problem Statement)**:
- Explicitly map each solution component to a problem from Slide 2
- Use phrases like: "To address Issue 1 (Multi-View Integration), we..."

**To Slide 4 (Implementation)**:
- Preview: "In the next slide, we'll show how we actually built this..."
- Set expectation: "Now you know WHAT we built, next is HOW we built it"

---

**End of Slide 3 Content**
