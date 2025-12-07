# Slide 2: Problem Statement

## 🎯 **Core Problem Definition**

### **The Central Challenge**
> *"How can we automatically generate natural language summaries of source code that are **structurally accurate**, **dependency-complete**, and **repository-aware**, while minimizing hallucinations and maximizing interpretability?"*

---

## 📐 **Formal Problem Statement**

### **Traditional Code Summarization (Insufficient)**

**Given:** A source code function `f`

**Goal:** Generate a natural language summary `s`

**Limitation:** Treats code as a **flat entity** (token sequence), ignoring:
- ❌ Structural representations (graphs)
- ❌ Repository context (cross-file dependencies)
- ❌ Quality criteria (what makes a summary "good")
- ❌ Dependency relationships (inter-procedural connections)

---

### **Our Enhanced Problem Formulation**

**Given:**
- 🗂️ **Repository** `R` containing `n` functions
- 🎯 **Target function** `f_i` to summarize
- 📊 **Structural graphs** `G(f_i)`:
  - `G_AST` - Abstract Syntax Tree
  - `G_CFG` - Control Flow Graph
  - `G_PDG` - Program Dependence Graph
  - `G_CG` - Call Graph (repository-wide)
- 🔗 **Context functions** `C(f_i, R)` - relevant callers and callees

**Goal:** Generate summary `s_i` that satisfies:

```
s_i = Summarize(f_i, Context(f_i, R), Graphs(f_i))
```

---

## 🎯 **Quality Criteria: What Makes a Good Summary?**

A generated summary **MUST** satisfy four key criteria:

### **1. Structural Accuracy** ✅
- Correctly describes **control flow** (loops, conditionals, error handling)
- Accurately represents **data dependencies** (which variables affect which computations)
- Reflects **execution semantics** encoded in graphs

**Example:**
```
❌ Bad: "Processes user data"
✅ Good: "Validates user input, then iterates through records in a try-catch block, 
         logging errors to the database on exception"
```

---

### **2. Dependency Completeness** 🔗
Summary must explicitly mention:

- **"Called by"**: Which functions invoke this one?
  - `Callers(f_i) = {f_j ∈ R | (f_j, f_i) ∈ CallGraph}`

- **"Calls"**: Which functions does this one depend on?
  - `Callees(f_i) = {f_k ∈ R | (f_i, f_k) ∈ CallGraph}`

**Example:**
```
❌ Bad: "Encrypts password using bcrypt"
✅ Good: "Encrypts password using bcrypt. Called by register_user() and 
         reset_password(). Calls hash_password() and validate_strength()"
```

---

### **3. Factual Consistency** 🛡️
- **No hallucinations**: Every claim must be derivable from code
- **No invented information**: Don't assume behavior not present in source
- **Verifiable against source**: Can trace every statement back to code

**Example Hallucination:**
```
Code: Uses bcrypt.hashpw()
❌ Hallucinated: "Uses AES-256 encryption for security"
✅ Factual: "Uses bcrypt hashing algorithm for password security"
```

---

### **4. Natural Language Quality** 📝
- **Fluent**: Grammatically correct, readable
- **Coherent**: Logical flow of information
- **Understandable**: Accessible to human developers
- **Concise**: No unnecessary verbosity

---

## 🚨 **Specific Issues to Address**

### **Issue 1: Multi-View Graph Integration Challenge**

**Problem:** How to integrate **four distinct graph representations** into a single coherent prompt without exceeding context window limits?

**Difficulty:**
- Each graph captures different information:
  - **AST**: Syntactic structure
  - **CFG**: Control flow paths
  - **PDG**: Data dependencies
  - **Call Graph**: Inter-procedural relationships
- Naive concatenation → **exceeds token budget** (4,096 tokens)
- Different graph structures (tree vs. directed graph vs. cyclic graph)

**Why Existing Approaches Fail:**
| Approach | Limitation |
|----------|------------|
| **GNN-based** (HAConvGNN) | Encode in embeddings → **lose interpretability** |
| **Pre-trained** (GraphCodeBERT) | Use only **one graph type** (data flow) |
| **Sequence-based** (CodeT5) | **Ignore structure** entirely |

**Our Challenge:** Serialize graphs into structured text while preserving critical information

---

### **Issue 2: Scalable Repository-Wide Context Extraction**

**Problem:** How to extract relevant context from **large repositories** (1,000+ files) while staying within context limits?

**Difficulty:**
- Call graph construction: **O(n)** complexity (n = number of files)
- Cross-file dependency resolution requires import analysis
- Relevant context may span **multiple files and directories**
- **Trade-off**: Completeness (all dependencies) vs. Conciseness (fit in context)

**Context Selection Problem:**
```
Given: Repository R with |R| = n functions
       Target function f_i
       Context window limit L (max tokens)

Find: Optimal subset C*(f_i, R) ⊆ C(f_i, R)

Maximize: Relevance(C*(f_i, R), f_i)
Subject to: |Prompt(f_i, C*(f_i, R))| ≤ L
```

**Relevance Scoring:**
- **Proximity**: Functions closer in call graph
- **Complexity**: Functions with higher cyclomatic complexity
- **Control Flow Importance**: Functions called in loops/conditionals

**Why Existing Approaches Fail:**
- **Function-level methods**: Ignore repository context entirely
- **Semantic retrieval (RAG)**: Use similarity, not **actual dependencies**
- **Context integration**: Only use nearby code, **miss distant dependencies**

---

### **Issue 3: Hallucination Mitigation**

**Problem:** How to prevent LLMs from generating plausible but **factually incorrect** summaries?

**Difficulty:**
- LLMs trained to generate fluent text → may "fill in" missing information
- No built-in verification mechanism
- Hard to distinguish confident correct vs. confident incorrect predictions
- Subtle hallucinations (e.g., wrong algorithm names, incorrect parameter types)

**Real-World Impact:**
- **Safety-critical systems**: Medical devices, autonomous vehicles
- **Security applications**: Cryptography, authentication
- **Financial systems**: Transaction processing, risk calculation

**Why Existing Approaches Fail:**
- **Single-pass generation**: No verification or self-correction
- **Pre-trained models**: Rely on implicit learning, no explicit verification
- **Ensemble methods**: May average errors but don't verify facts

**Our Challenge:** Design agentic workflow with explicit critique and verification

---

### **Issue 4: Explicit Dependency Extraction**

**Problem:** How to reliably extract and present "Called by" and "Calls" relationships?

**Difficulty:**
- Must handle:
  - ✓ Direct function calls
  - ✓ Method calls on objects
  - ✓ Imported functions from other modules
  - ✓ Dynamic calls (`getattr`, callbacks, decorators)
- Cross-file resolution requires import analysis
- LLMs may **omit dependency info** even when provided in prompt
- Need to verify LLM actually includes dependencies in output

**Example Complexity:**
```python
# Direct call
result = process_data(input)

# Method call
user.authenticate(password)

# Imported function
from utils.crypto import encrypt
encrypted = encrypt(data)

# Dynamic call (hard to resolve statically)
method = getattr(obj, method_name)
method()
```

**Why Existing Approaches Fail:**
- **All 2021 approaches**: Don't generate explicit dependency information
- **HA-ConvGNN**: Uses call graph but encodes in embeddings (not text)
- **Context integration**: Includes surrounding code but doesn't extract explicit dependencies

---

### **Issue 5: Balancing Interpretability vs. Performance**

**Problem:** How to maintain **interpretability** (explicit prompts) while achieving **competitive performance** with black-box models?

**Trade-off:**
```
┌─────────────────────────────────────────┐
│  High Performance  │  High Interpretability │
│  (GNN embeddings)  │  (Explicit prompts)    │
├────────────────────┼────────────────────────┤
│  ✅ Learn complex   │  ✅ Human-readable      │
│     patterns       │  ✅ Debuggable          │
│  ✅ State-of-art    │  ✅ Verifiable          │
│     metrics        │                        │
│  ❌ Black-box       │  ❌ May lose some       │
│  ❌ Not debuggable  │     expressiveness     │
└─────────────────────────────────────────┘
```

**Our Challenge:** Prove that explicit structural prompts can match or exceed GNN performance

---

## 📊 **Problem Scope & Constraints**

### **In Scope** ✅
- Python code exclusively
- Function-level and method-level summarization
- Four graph types (AST, CFG, PDG, Call Graph)
- Repository-wide dependency resolution
- Automated evaluation metrics + human evaluation

### **Out of Scope** ❌
- Multi-language support (Java, C++, JavaScript)
- Dynamic analysis (runtime behavior)
- Production deployment features (REST API, cloud)
- Repositories > 10,000 files (performance limitations)

### **Key Constraints**
1. **Context Window**: ≤ 4,096 tokens (Gemma-2b limit)
2. **Repository Size**: Optimized for ≤ 1,000 files
3. **Inference Time**: Target ≤ 10 seconds per function
4. **Model Size**: Gemma-2b (2B parameters) for accessibility
5. **Training Data**: Limited dependency-rich examples (386 custom + CodeXGlue augmentation)

---

## 🎯 **Research Questions**

Based on these issues, our research addresses:

1. **RQ1**: Can explicit multi-view structural prompts match the performance of implicit GNN embeddings?

2. **RQ2**: Does repository-wide context significantly improve summary quality compared to function-level analysis?

3. **RQ3**: Can an agentic critique-and-refine workflow reduce hallucinations in code summarization?

4. **RQ4**: What is the optimal balance between structural detail and context window usage?

5. **RQ5**: How can we automatically evaluate dependency completeness in generated summaries?

---

## 🔄 **Problem Visualization**

### **The Gap We're Filling**

```
Current State                    Desired State
─────────────                    ─────────────
Function f                       Function f
    ↓                                ↓
[Tokenize]                       [Multi-Graph Analysis]
    ↓                                ↓
"def process..."          AST + CFG + PDG + CallGraph
    ↓                                ↓
[LLM Generate]                   [Structured Prompt]
    ↓                                ↓
"Processes data"          [LLM with Context + Critique]
                                     ↓
❌ Generic               ✅ "Validates input, iterates with
❌ No dependencies          error handling. Called by
❌ May hallucinate          register_user(). Calls
❌ No verification          validate_strength() and
                            hash_password()"
```

---

## 💡 **Key Takeaway**

The problem is **not just** generating summaries—it's generating summaries that:
1. ✅ Reflect **deep structural understanding** (4 graph types)
2. ✅ Include **repository-wide context** (cross-file dependencies)
3. ✅ Are **factually accurate** (no hallucinations)
4. ✅ Explicitly mention **dependencies** ("Called by", "Calls")
5. ✅ Remain **interpretable** (not black-box embeddings)

**This requires a fundamentally different approach than existing methods.**

---

## 🎤 **Transition to Next Slide**

"Now that we've clearly defined the problem and identified the specific technical challenges, let's examine what existing research has attempted and why those approaches fall short..."

---

## 📝 **Speaker Notes**

### **Opening (30 seconds)**
- Start with the formal problem statement (shows rigor)
- Immediately contrast traditional vs. enhanced formulation
- Emphasize the **four quality criteria** as your evaluation framework

### **Quality Criteria Section (1.5 minutes)**
- Use the **examples** liberally—they make abstract concepts concrete
- The hallucination example (AES vs. bcrypt) is particularly memorable
- Emphasize that **all four criteria must be met** (not just fluency)

### **Specific Issues Section (3 minutes)**
- This is the **core** of the slide—spend the most time here
- For each issue:
  1. State the problem clearly
  2. Explain why it's difficult
  3. Show why existing approaches fail
  4. Preview your approach (without details yet)
- Use the **comparison tables** to create visual contrast

### **Research Questions (1 minute)**
- Frame these as **testable hypotheses**
- Connect each RQ back to a specific issue
- This shows your work is **scientifically rigorous**

### **Key Messages to Emphasize**
1. **This is a well-defined, formal problem** (not vague exploration)
2. **Multiple interconnected challenges** (not just one issue)
3. **Existing solutions are fundamentally limited** (not just need tuning)
4. **Clear success criteria** (the four quality requirements)

### **Anticipated Questions**
- *Q: Why focus on Python only?*
  - A: Scoping for feasibility; methodology generalizes to other languages
- *Q: Why 4,096 token limit?*
  - A: Gemma-2b model constraint; balances accessibility with capability
- *Q: How do you measure "interpretability"?*
  - A: Human ability to trace summary claims back to structural prompts
- *Q: What if the repository is larger than 1,000 files?*
  - A: Intelligent subgraph extraction; focus on relevant context

### **Common Pitfalls to Avoid**
- ❌ Don't get too mathematical (save formal definitions for thesis)
- ❌ Don't list issues without explaining **why** they're hard
- ❌ Don't criticize existing work without being constructive
- ✅ **Do** use concrete examples for every abstract concept
- ✅ **Do** connect back to real-world developer pain points

---

## 🎨 **Visual Suggestions**

### **Slide Layout**
- **Problem Statement Box** at top (highlighted, large font)
- **Four Quality Criteria** as quadrants or pillars
- **Five Issues** as numbered sections with icons
- **Visualization** showing current state vs. desired state

### **Color Coding**
- 🔴 **Red**: Problems and limitations
- 🟡 **Yellow**: Challenges and difficulties
- 🟢 **Green**: Our approach/solution (preview)
- 🔵 **Blue**: Formal definitions and constraints

### **Icons/Visuals**
- 📊 **Graphs**: Show AST, CFG, PDG, Call Graph as small diagrams
- 🔗 **Dependencies**: Use arrows to show "Called by" and "Calls"
- ⚠️ **Hallucination**: Use warning symbol for factual consistency
- 🎯 **Target**: Use for quality criteria

### **Diagrams to Include**
1. **Traditional vs. Enhanced Formulation** (side-by-side comparison)
2. **Four Quality Criteria** (visual checklist or pillars)
3. **Context Window Challenge** (graph → text serialization)
4. **Current State vs. Desired State** (before/after flowchart)

---

## 📚 **Key Technical Terms to Define**

Make sure your audience understands:
- **AST** (Abstract Syntax Tree): Syntactic structure
- **CFG** (Control Flow Graph): Execution paths
- **PDG** (Program Dependence Graph): Data dependencies
- **Call Graph**: Function invocation relationships
- **Hallucination**: Generating plausible but incorrect information
- **Context Window**: Maximum input tokens for LLM

*Consider having a "terminology" backup slide*

---

## 🔗 **Connection to Previous Slide**

**Slide 1** established **why** this matters (motivation)
**Slide 2** defines **what** the problem is (formal statement)
**Slide 3** will show **what others have tried** (related work)

This creates a logical narrative flow.

---

## ⏱️ **Timing Recommendation**

- **Total time**: 5-6 minutes
- **Problem definition**: 1 minute
- **Quality criteria**: 1.5 minutes
- **Five specific issues**: 2.5 minutes (30 seconds each)
- **Research questions**: 1 minute

Adjust based on your total presentation time allocation.

---

**End of Slide 2 Content**
