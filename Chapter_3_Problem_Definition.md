# Chapter 3: Problem Definition

## 3.1 Introduction

This chapter provides a rigorous formulation of the code summarization problem that NeuroGraph-CodeRAG addresses. We begin with a formal mathematical definition of the problem, followed by a detailed specification of assumptions, scope, and constraints. We then identify and analyze the major technical challenges that make this problem difficult, explaining why existing approaches (as reviewed in Chapter 2) have been insufficient.

The problem we address is fundamentally different from traditional code summarization in three key aspects: (1) we require **explicit structural reasoning** over multiple program representations, (2) we demand **repository-wide context awareness** rather than function-level analysis, and (3) we aim to generate **dependency-rich summaries** that explicitly detail inter-procedural relationships. These requirements introduce unique challenges that necessitate novel solutions.

---

## 3.2 Formal Problem Statement

### 3.2.1 Basic Code Summarization Problem

**Definition 3.1 (Traditional Code Summarization)**

Given a source code function $f \in \mathcal{F}$, where $\mathcal{F}$ is the space of all valid functions in a programming language $L$, the code summarization task is to generate a natural language summary $s \in \mathcal{S}$, where $\mathcal{S}$ is the space of all possible natural language descriptions.

Formally, we seek a function:

$$\text{Summarize}: \mathcal{F} \rightarrow \mathcal{S}$$

such that $s = \text{Summarize}(f)$ accurately describes the functionality, behavior, and purpose of $f$.

**Limitations of Traditional Formulation**

This formulation has several critical limitations:
1. **No structural representation**: Treats $f$ as a flat entity (typically a token sequence)
2. **No repository context**: Function $f$ analyzed in isolation
3. **No quality criteria**: Doesn't specify what makes a summary "accurate"
4. **No dependency information**: Doesn't require inter-procedural relationships

---

### 3.2.2 Enhanced Problem Formulation

**Definition 3.2 (Graph-Augmented Repository-Aware Code Summarization)**

Let:
- $R = \{f_1, f_2, \ldots, f_n\}$ be a code repository containing $n$ functions
- $f_i \in R$ be the target function to summarize
- $\mathcal{G}(f_i) = \{G_{\text{AST}}, G_{\text{CFG}}, G_{\text{PDG}}, G_{\text{CG}}\}$ be the set of structural graphs for $f_i$:
  - $G_{\text{AST}}$: Abstract Syntax Tree
  - $G_{\text{CFG}}$: Control Flow Graph
  - $G_{\text{PDG}}$: Program Dependence Graph
  - $G_{\text{CG}}$: Call Graph (repository-wide)
- $C(f_i, R) \subseteq R$ be the relevant context functions (callees and callers of $f_i$)
- $s_i$ be the desired summary for $f_i$

The **graph-augmented repository-aware code summarization** task is to learn a function:

$$\text{Summarize}_{\text{GAR}}: \mathcal{F} \times \mathcal{P}(R) \times \mathcal{G} \rightarrow \mathcal{S}$$

where $\mathcal{P}(R)$ is the power set of $R$ (all possible subsets representing context), such that:

$$s_i = \text{Summarize}_{\text{GAR}}(f_i, C(f_i, R), \mathcal{G}(f_i))$$

**Quality Criteria**

The generated summary $s_i$ must satisfy:

1. **Structural Accuracy**: $s_i$ correctly describes the control flow, data dependencies, and execution semantics encoded in $\mathcal{G}(f_i)$

2. **Dependency Completeness**: $s_i$ explicitly mentions:
   - **Called by**: $\text{Callers}(f_i) = \{f_j \in R \mid (f_j, f_i) \in E_{CG}\}$
   - **Calls**: $\text{Callees}(f_i) = \{f_k \in R \mid (f_i, f_k) \in E_{CG}\}$
   
   where $E_{CG}$ is the edge set of the call graph $G_{\text{CG}}$

3. **Factual Consistency**: $s_i$ contains no hallucinated information not derivable from $f_i$, $C(f_i, R)$, or $\mathcal{G}(f_i)$

4. **Natural Language Quality**: $s_i$ is fluent, coherent, and understandable to human developers

---

### 3.2.3 Multi-View Structural Representation

**Definition 3.3 (Structural Graph Ensemble)**

For a function $f_i$, the structural graph ensemble $\mathcal{G}(f_i)$ consists of:

**1. Abstract Syntax Tree (AST)**

$$G_{\text{AST}} = (V_{\text{AST}}, E_{\text{AST}})$$

where:
- $V_{\text{AST}}$: nodes representing syntactic constructs (statements, expressions, identifiers)
- $E_{\text{AST}}$: parent-child relationships in the syntax tree

**2. Control Flow Graph (CFG)**

$$G_{\text{CFG}} = (V_{\text{CFG}}, E_{\text{CFG}})$$

where:
- $V_{\text{CFG}}$: basic blocks of code (sequences of statements with single entry/exit)
- $E_{\text{CFG}}$: control flow edges representing possible execution paths
- Each edge $e \in E_{\text{CFG}}$ may be labeled with conditions (for branches)

**3. Program Dependence Graph (PDG)**

$$G_{\text{PDG}} = (V_{\text{PDG}}, E_{\text{data}} \cup E_{\text{control}})$$

where:
- $V_{\text{PDG}}$: program statements and expressions
- $E_{\text{data}}$: data dependency edges $(v_i, v_j)$ indicating $v_j$ uses a value defined by $v_i$
- $E_{\text{control}}$: control dependency edges $(v_i, v_j)$ indicating $v_j$'s execution depends on $v_i$

**4. Call Graph (CG)**

$$G_{\text{CG}} = (V_{\text{CG}}, E_{\text{CG}})$$

where:
- $V_{\text{CG}} = R$: all functions in the repository
- $E_{\text{CG}}$: edges $(f_i, f_j)$ indicating $f_i$ calls $f_j$

**Structural Prompt Construction**

The structural information is transformed into a textual prompt $P(f_i, \mathcal{G}(f_i))$ that explicitly presents:

$$P(f_i, \mathcal{G}(f_i)) = \text{Serialize}(\mathcal{G}(f_i)) \oplus \text{Code}(f_i)$$

where $\oplus$ denotes concatenation and $\text{Serialize}$ converts graph structures to human-readable text.

---

### 3.2.4 Repository Context Extraction

**Definition 3.4 (Relevant Context Set)**

For a target function $f_i$ in repository $R$, the relevant context set $C(f_i, R)$ is defined as:

$$C(f_i, R) = \text{Callees}(f_i) \cup \text{Callers}(f_i) \cup \text{TransitiveDeps}(f_i, k)$$

where:
- $\text{Callees}(f_i)$: functions directly called by $f_i$
- $\text{Callers}(f_i)$: functions that directly call $f_i$
- $\text{TransitiveDeps}(f_i, k)$: functions within $k$ hops in the call graph

**Subgraph Extraction Problem**

Given:
- Repository $R$ with $|R| = n$ functions
- Target function $f_i$
- Context window limit $L$ (maximum tokens)

Find: $C^*(f_i, R) \subseteq C(f_i, R)$ such that:

$$\text{maximize} \quad \text{Relevance}(C^*(f_i, R), f_i)$$

$$\text{subject to} \quad |P(f_i, C^*(f_i, R))| \leq L$$

where $\text{Relevance}$ is a scoring function that prioritizes:
1. **Proximity**: Functions closer in the call graph
2. **Complexity**: Functions with higher cyclomatic complexity (more logic)
3. **Control Flow Importance**: Functions called within loops or conditionals

**Relevance Scoring Function**

$$\text{Relevance}(f_j, f_i) = \alpha \cdot \text{Proximity}(f_j, f_i) + \beta \cdot \text{Complexity}(f_j) + \gamma \cdot \text{CFImportance}(f_j, f_i)$$

where $\alpha, \beta, \gamma$ are weighting parameters.

---

### 3.2.5 Agentic Refinement Problem

**Definition 3.5 (Iterative Summary Refinement)**

Given:
- Initial summary $s_0 = \text{Generate}(f_i, C(f_i, R), \mathcal{G}(f_i))$
- Critique function $\text{Critique}: \mathcal{S} \times \mathcal{F} \rightarrow \mathcal{C}$ that identifies errors/gaps
- Repository graph $G_{\text{CG}}$ for context consultation
- Maximum iterations $T$

The agentic refinement process produces a sequence of summaries:

$$s_0, s_1, s_2, \ldots, s_t$$

where each iteration $t$:

1. **Critique**: $c_t = \text{Critique}(s_{t-1}, f_i)$
2. **Decision**: 
   $$a_t = \begin{cases}
   \text{CONSULT} & \text{if } c_t \text{ identifies missing dependencies} \\
   \text{REFINE} & \text{if } c_t \text{ identifies logical errors} \\
   \text{FINISH} & \text{if } \text{Quality}(s_{t-1}) > \theta
   \end{cases}$$
3. **Action**:
   - If $a_t = \text{CONSULT}$: Retrieve missing functions from $G_{\text{CG}}$, update context $C'(f_i, R)$
   - If $a_t = \text{REFINE}$: $s_t = \text{Refine}(s_{t-1}, c_t)$
   - If $a_t = \text{FINISH}$: Return $s_{t-1}$

**Objective**: Minimize hallucinations and maximize factual accuracy through iterative verification and refinement.

---

### 3.2.6 Optimization Objective

**Definition 3.6 (Overall Optimization Problem)**

Given a dataset $\mathcal{D} = \{(f_i, R_i, s_i^*)\}_{i=1}^N$ where $s_i^*$ is the ground-truth summary, we seek to learn parameters $\theta$ for the summarization system that minimize:

$$\mathcal{L}(\theta) = \sum_{i=1}^N \left[ \mathcal{L}_{\text{gen}}(s_i, s_i^*) + \lambda_1 \mathcal{L}_{\text{struct}}(s_i, \mathcal{G}(f_i)) + \lambda_2 \mathcal{L}_{\text{dep}}(s_i, C(f_i, R_i)) \right]$$

where:
- $\mathcal{L}_{\text{gen}}$: Standard generation loss (e.g., cross-entropy for token prediction)
- $\mathcal{L}_{\text{struct}}$: Structural accuracy loss (penalizes summaries that misrepresent control flow/data dependencies)
- $\mathcal{L}_{\text{dep}}$: Dependency coverage loss (penalizes missing "Called by"/"Calls" information)
- $\lambda_1, \lambda_2$: Weighting hyperparameters

---

## 3.3 Assumptions

### 3.3.1 Code Assumptions

**A1: Well-Formed Code**
- **Assumption**: Input code $f \in \mathcal{F}$ is syntactically valid and parseable by standard Python AST tools
- **Justification**: Enables reliable structural analysis; malformed code would require error recovery mechanisms beyond scope
- **Limitation**: Cannot handle obfuscated or intentionally malformed code

**A2: Reasonable Naming Conventions**
- **Assumption**: Identifiers (variable names, function names) follow conventional naming practices
- **Justification**: Identifier names provide semantic signals; heavily obfuscated code (e.g., `a`, `b`, `c`) loses this information
- **Limitation**: Performance may degrade on code with meaningless identifier names

**A3: Standard Repository Structure**
- **Assumption**: Code repositories follow common Python project layouts (e.g., `src/`, `lib/`, `__init__.py` for packages)
- **Justification**: Enables cross-file import resolution and dependency tracking
- **Limitation**: Non-standard structures may require manual configuration

**A4: Static Analyzability**
- **Assumption**: Function behavior is determinable through static analysis
- **Justification**: Our approach uses static analysis (AST, CFG, PDG); dynamic behavior not captured
- **Limitation**: Cannot capture runtime-dependent behavior (e.g., reflection, dynamic imports, external API calls)

---

### 3.3.2 Resource Assumptions

**A5: Computational Resources**
- **Assumption**: Access to GPU for model inference (recommended but not required)
- **Justification**: LLM inference is computationally intensive; GPU significantly speeds up generation
- **Limitation**: CPU-only execution possible but slower (5-10x)

**A6: Memory Availability**
- **Assumption**: Sufficient memory to load model (≥8GB RAM) and construct repository graphs
- **Justification**: Gemma-2b requires ~4GB; graph construction for large repos requires additional memory
- **Limitation**: Very large repositories (>10,000 files) may exceed memory limits

---

### 3.3.3 Data Assumptions

**A7: Training Data Quality**
- **Assumption**: Training dataset contains high-quality code-summary pairs with dependency-rich summaries
- **Justification**: Model learns from examples; poor-quality training data yields poor summaries
- **Limitation**: Requires curated dataset; automatically scraped data may be noisy

**A8: Ground Truth Availability**
- **Assumption**: For evaluation, ground-truth summaries are available
- **Justification**: Enables quantitative evaluation using automated metrics (BLEU, ROUGE)
- **Limitation**: Human-written summaries may have variability; single reference may not capture all valid summaries

---

### 3.3.4 Model Assumptions

**A9: LLM Instruction Following**
- **Assumption**: Base LLM (Gemma-2b) can follow structured prompts and generate coherent summaries
- **Justification**: Modern LLMs demonstrate strong instruction-following capabilities
- **Limitation**: Model may occasionally ignore prompt instructions or generate off-topic text

**A10: Context Window Sufficiency**
- **Assumption**: Relevant structural information and context fit within model's context window (~4,096 tokens)
- **Justification**: Intelligent subgraph extraction prioritizes most relevant context
- **Limitation**: Extremely large functions or extensive dependency chains may require truncation

---

## 3.4 Scope and Constraints

### 3.4.1 Scope

**In Scope:**

1. **Language Support**
   - Python code exclusively
   - Function-level and method-level summarization
   - Class-level analysis for context

2. **Structural Analysis**
   - Abstract Syntax Tree (AST) extraction
   - Control Flow Graph (CFG) construction
   - Program Dependence Graph (PDG) generation
   - Inter-procedural Call Graph building

3. **Repository Analysis**
   - Cross-file dependency resolution
   - Import statement analysis
   - Repository-wide call graph construction
   - Intelligent subgraph extraction

4. **Summary Generation**
   - Natural language descriptions of functionality
   - Explicit "Called by" and "Calls" information
   - Control flow and data dependency explanations
   - Parameter and return value descriptions

5. **Evaluation**
   - Automated metrics (BLEU, ROUGE, METEOR, BERTScore)
   - Structural accuracy validation
   - Dependency coverage analysis
   - Qualitative human evaluation (sample-based)

**Out of Scope:**

1. **Multi-Language Support**
   - Languages other than Python (Java, C++, JavaScript) not supported
   - Requires language-specific parsers and analysis tools

2. **Dynamic Analysis**
   - Runtime behavior not captured
   - No execution tracing or profiling
   - Cannot analyze behavior dependent on external systems

3. **Production Features**
   - No REST API endpoints
   - No cloud deployment
   - No multi-user authentication/authorization
   - No continuous integration/deployment

4. **Large-Scale Optimization**
   - Repositories >10,000 files may have performance issues
   - No distributed processing for batch summarization
   - No advanced caching strategies

5. **Interactive Features**
   - No user-guided summary refinement
   - No interactive query answering
   - No incremental summarization

---

### 3.4.2 Constraints

**C1: Context Window Constraint**
- **Constraint**: Total prompt length $|P| \leq L$ where $L \approx 4,096$ tokens
- **Impact**: Limits amount of structural information and context that can be included
- **Mitigation**: Intelligent subgraph extraction prioritizes most relevant context

**C2: Repository Size Constraint**
- **Constraint**: Optimized for repositories with $|R| \leq 1,000$ files
- **Impact**: Graph construction time and memory usage scale with repository size
- **Mitigation**: Incremental graph construction; caching of parsed structures

**C3: Inference Time Constraint**
- **Constraint**: Target inference time $\leq 10$ seconds per function on consumer hardware
- **Impact**: Limits number of agentic refinement iterations
- **Mitigation**: Early stopping when quality threshold met; maximum iteration limit

**C4: Model Size Constraint**
- **Constraint**: Use Gemma-2b (2 billion parameters) for accessibility
- **Impact**: Smaller model may have lower fluency than GPT-4 or Claude
- **Mitigation**: Fine-tuning with LoRA; explicit structural prompts compensate for smaller size

**C5: Training Data Constraint**
- **Constraint**: Limited availability of high-quality dependency-rich summaries
- **Impact**: Custom dataset of 386 examples may be insufficient for full generalization
- **Mitigation**: Augmentation with CodeXGlue dataset; synthetic data generation

**C6: Evaluation Resource Constraint**
- **Constraint**: Human evaluation limited to sample sets (not exhaustive)
- **Impact**: Cannot evaluate all generated summaries manually
- **Mitigation**: Stratified sampling; focus on diverse code patterns

---

## 3.5 Challenges

### 3.5.1 Technical Challenges

#### **Challenge 1: Multi-View Graph Integration**

**Problem**: How to effectively integrate four distinct graph representations (AST, CFG, PDG, Call Graph) into a single coherent prompt without overwhelming the LLM's context window?

**Difficulty**:
- Each graph type captures different information (syntax, control flow, data dependencies, inter-procedural calls)
- Naive concatenation of all graphs exceeds context window limits
- Graphs have different structures (tree vs. directed graph vs. cyclic graph)
- Need to preserve critical information while staying within token budget

**Why Existing Approaches Fail**:
- **GNN-based methods** (HAConvGNN, 2021): Encode graphs in learned embeddings, losing interpretability
- **Pre-trained models** (GraphCodeBERT, 2021): Use only one graph type (data flow), miss others
- **Sequence-based models** (CodeT5, 2021): Ignore structure entirely

**Our Approach**:
- Serialize graphs into structured textual format
- Prioritize critical structural elements (loops, conditionals, complex data flows)
- Use hierarchical representation (summary statistics + detailed critical paths)

**Open Questions**:
- What is the optimal balance between structural detail and context window usage?
- Which structural features are most informative for summary generation?
- How to automatically determine which graph elements to prioritize?

---

#### **Challenge 2: Scalable Repository-Wide Context Extraction**

**Problem**: How to efficiently extract relevant context from large repositories (potentially thousands of files) while maintaining accuracy and staying within context window limits?

**Difficulty**:
- Call graph construction requires parsing entire repository ($O(n)$ where $n$ = number of files)
- Cross-file dependency resolution requires import analysis and symbol resolution
- Relevant context may span multiple files and directories
- Context window limits require selecting subset of relevant functions
- Trade-off between completeness (include all dependencies) and conciseness (fit in context window)

**Why Existing Approaches Fail**:
- **Function-level methods** (all 2021 approaches): Ignore repository context entirely
- **Semantic retrieval** (RAG systems): Use similarity, not actual dependencies
- **Context integration** (Lin et al., 2021): Only use spatially proximate code, miss distant dependencies

**Our Approach**:
- Build global call graph once, reuse for multiple queries
- Intelligent subgraph extraction with relevance scoring
- Prioritize based on proximity, complexity, and control flow importance

**Open Questions**:
- How to efficiently update call graph when repository changes?
- What is the optimal relevance scoring function?
- How to handle indirect dependencies (transitive calls)?

---

#### **Challenge 3: Hallucination Mitigation in Generative Models**

**Problem**: How to prevent LLMs from generating plausible-sounding but factually incorrect summaries, especially when encountering unfamiliar code patterns or missing context?

**Difficulty**:
- LLMs are trained to generate fluent text, may "fill in" missing information
- No built-in mechanism to verify factual accuracy against source code
- Difficult to distinguish confident correct predictions from confident incorrect ones
- Hallucinations can be subtle (e.g., claiming function uses AES encryption when it uses bcrypt)

**Why Existing Approaches Fail**:
- **Single-pass generation** (all 2021 approaches): No verification or self-correction
- **Pre-trained models** (GraphCodeBERT, CodeT5): Rely on implicit learning, no explicit verification
- **Ensemble methods** (LeClair et al., 2021): May average out errors but don't verify facts

**Our Approach**:
- Agentic workflow with explicit critique step
- Critique LLM compares summary against source code
- Repository graph consultation when missing information identified
- Iterative refinement until quality threshold met

**Open Questions**:
- How to design effective critique prompts?
- When to consult repository graph vs. refine with existing information?
- How many refinement iterations are optimal?

---

#### **Challenge 4: Explicit Dependency Extraction and Presentation**

**Problem**: How to reliably extract "Called by" and "Calls" relationships from code and ensure they are explicitly mentioned in generated summaries?

**Difficulty**:
- Call graph construction requires handling:
  - Direct function calls
  - Method calls on objects
  - Imported functions from other modules
  - Dynamic calls (e.g., `getattr`, callbacks)
- Cross-file resolution requires import analysis
- Generated summaries may omit dependency information even when provided in prompt
- Need to verify that LLM actually includes dependency information in output

**Why Existing Approaches Fail**:
- **All 2021 approaches**: Do not explicitly generate dependency information in summaries
- **HA-ConvGNN** (Liu et al., 2021): Uses call graph but encodes in embeddings, doesn't generate explicit text
- **Context integration** (Lin et al., 2021): Includes surrounding code but doesn't extract explicit dependencies

**Our Approach**:
- Explicit call graph construction with cross-file resolution
- Structured prompt section dedicated to dependencies
- Critique step verifies dependency information is present in summary
- Fine-tuning on dataset with dependency-rich summaries

**Open Questions**:
- How to handle dynamic calls that cannot be statically resolved?
- How to present dependency information without making summaries verbose?
- How to prioritize which dependencies to mention (if many)?

---

#### **Challenge 5: Balancing Interpretability and Performance**

**Problem**: How to maintain interpretability (explicit structural prompts) while achieving competitive performance with black-box neural models?

**Difficulty**:
- GNN-based models can learn complex structural patterns through embeddings
- Explicit textual prompts may be less expressive than learned embeddings
- Serializing graphs to text loses some structural information (e.g., graph topology)
- Trade-off between human-readability and model performance

**Why Existing Approaches Fail**:
- **GNN methods** (HAConvGNN, Retrieval-Augmented GNN): High performance but opaque
- **Pre-trained models** (GraphCodeBERT): Implicit structure, no interpretability
- **Template-based methods**: Interpretable but poor performance

**Our Approach**:
- Leverage LLM's in-context learning with structured prompts
- Design prompt format that balances readability and information density
- Use LoRA fine-tuning to adapt model to structural prompts
- Empirically validate that explicit prompts achieve competitive performance

**Open Questions**:
- Can explicit prompts match or exceed GNN performance?
- What is the optimal prompt structure for structural information?
- How to quantify interpretability vs. performance trade-off?

---

### 3.5.2 Methodological Challenges

#### **Challenge 6: Evaluation Metrics for Dependency-Rich Summaries**

**Problem**: How to evaluate whether generated summaries correctly include dependency information, beyond standard BLEU/ROUGE metrics?

**Difficulty**:
- BLEU/ROUGE measure token overlap, not semantic correctness
- A summary can have high BLEU but incorrect dependency information
- Need metrics that specifically evaluate:
  - Presence of "Called by" information
  - Accuracy of "Calls" information
  - Correctness of structural descriptions (control flow, data dependencies)

**Why Existing Approaches Fail**:
- **Standard metrics** (BLEU, ROUGE): Don't capture structural or dependency accuracy
- **Shi et al. (2021) survey**: Identified metric limitations but didn't propose alternatives
- **No existing metrics** for dependency coverage

**Our Approach**:
- Define dependency coverage metric: $\text{DepCov} = \frac{|\text{Deps mentioned in summary}|}{|\text{Actual deps}|}$
- Structural accuracy metric: Compare mentioned control flow patterns against actual CFG
- Human evaluation focused on dependency correctness

**Open Questions**:
- How to automatically verify dependency mentions in free-form text?
- How to weight different types of dependencies (direct calls vs. transitive)?
- How to handle partial correctness (e.g., mentions some but not all callees)?

---

#### **Challenge 7: Dataset Construction for Dependency-Rich Summaries**

**Problem**: How to create a training dataset with high-quality dependency-rich summaries when such data is scarce?

**Difficulty**:
- Existing datasets (CodeSearchNet, CodeXGlue) have generic summaries lacking dependency information
- Manually writing dependency-rich summaries is time-consuming (requires code analysis)
- Automatically generated summaries may have errors
- Need sufficient diversity in code patterns and dependency structures

**Why Existing Approaches Fail**:
- **CodeSearchNet/CodeXGlue**: Scraped from docstrings, often lack dependency details
- **Manual annotation**: Expensive and doesn't scale
- **Synthetic generation**: May introduce biases or errors

**Our Approach**:
- Curated custom dataset of 386 examples with manually verified dependency-rich summaries
- Augmentation with CodeXGlue dataset (400K+ examples) for general fluency
- Hybrid training: dependency-rich examples + large-scale generic examples

**Open Questions**:
- What is the minimum dataset size for effective learning?
- How to balance custom (high-quality, small) vs. large-scale (lower-quality, large) data?
- Can we automatically augment existing summaries with dependency information?

---

#### **Challenge 8: Reproducibility and Fair Comparison**

**Problem**: How to ensure reproducible results and fair comparison with existing approaches given the methodological issues identified by Shi et al. (2021)?

**Difficulty**:
- Different BLEU implementations produce different scores
- Pre-processing choices significantly impact performance
- Hyperparameter sensitivity
- Dataset characteristics (size, splitting, duplication) affect results

**Why Existing Approaches Fail**:
- **Shi et al. (2021)**: Identified problems but many papers still don't report full details
- **Inconsistent evaluation**: Different papers use different setups
- **Reproducibility crisis**: Difficulty reproducing reported results

**Our Approach**:
- Fully document all implementation details
- Use standardized BLEU implementation (SacreBLEU)
- Report all pre-processing steps
- Provide open-source code for reproducibility
- Conduct ablation studies to isolate contributions

**Open Questions**:
- How to fairly compare with methods that used different evaluation setups?
- How to account for improvements in base models (e.g., Gemma-2b vs. older models)?
- How to separate genuine algorithmic improvements from evaluation artifacts?

---

### 3.5.3 Practical Challenges

#### **Challenge 9: Computational Efficiency**

**Problem**: How to make the system practical for real-world use given the computational overhead of graph construction, LLM inference, and agentic iteration?

**Difficulty**:
- Repository graph construction: $O(n)$ where $n$ = number of files
- LLM inference: ~1-2 seconds per generation on GPU
- Agentic refinement: Multiple LLM calls per summary
- Total time per function: Graph construction + multiple LLM calls

**Trade-offs**:
- **Accuracy vs. Speed**: More iterations improve quality but increase time
- **Context vs. Efficiency**: More context improves accuracy but slows inference
- **Caching vs. Freshness**: Cached graphs fast but may be stale

**Our Approach**:
- One-time graph construction, reuse for multiple queries
- Early stopping in agentic loop when quality threshold met
- Maximum iteration limit (e.g., 3-5 iterations)
- Caching of parsed ASTs and CFGs

**Open Questions**:
- What is acceptable latency for developers?
- How to incrementally update graphs when code changes?
- Can we predict when agentic refinement will help vs. waste time?

---

#### **Challenge 10: Generalization to Unseen Code Patterns**

**Problem**: How to ensure the system generalizes to code patterns not seen during training?

**Difficulty**:
- Training data may not cover all possible:
  - Programming paradigms (OOP, functional, procedural)
  - Design patterns (singleton, factory, observer, etc.)
  - Domain-specific logic (ML, web, systems, etc.)
- Model may overfit to training distribution
- Structural prompts help but don't guarantee generalization

**Why Existing Approaches Fail**:
- **All neural models**: Risk of overfitting to training data
- **Limited diversity**: Most datasets focus on specific domains (e.g., web development)

**Our Approach**:
- Diverse training data covering multiple domains
- Structural prompts provide explicit grounding (less reliance on memorization)
- Agentic critique can identify when model is uncertain
- Evaluation on out-of-domain test sets

**Open Questions**:
- How to measure generalization beyond standard test sets?
- Can structural prompts fully compensate for limited training data?
- How to detect when model is operating outside its competence?

---

## 3.6 Summary

This chapter has provided a rigorous formulation of the graph-augmented repository-aware code summarization problem. Key contributions include:

1. **Formal Problem Statement**: Mathematical formulation extending traditional code summarization to include:
   - Multi-view structural graphs ($G_{\text{AST}}, G_{\text{CFG}}, G_{\text{PDG}}, G_{\text{CG}}$)
   - Repository-wide context $C(f_i, R)$
   - Dependency completeness requirements
   - Agentic refinement process

2. **Assumptions**: Clearly stated assumptions about:
   - Code quality and structure (A1-A4)
   - Computational resources (A5-A6)
   - Data availability (A7-A8)
   - Model capabilities (A9-A10)

3. **Scope and Constraints**: Explicit boundaries defining:
   - What is in scope (Python, structural analysis, repository-wide context)
   - What is out of scope (multi-language, dynamic analysis, production features)
   - Six key constraints (C1-C6) limiting the solution space

4. **Challenges**: Identified and analyzed 10 major challenges:
   - **Technical** (C1-C5): Multi-view integration, scalable context extraction, hallucination mitigation, dependency extraction, interpretability
   - **Methodological** (C6-C8): Evaluation metrics, dataset construction, reproducibility
   - **Practical** (C9-C10): Computational efficiency, generalization

For each challenge, we explained:
- Why it is difficult
- Why existing 2021 approaches fail to address it
- Our proposed approach
- Open research questions

The following chapters will describe how NeuroGraph-CodeRAG addresses these challenges through its system architecture (Chapter 4), implementation (Chapter 5), and experimental validation (Chapter 6).

---

**End of Chapter 3**
