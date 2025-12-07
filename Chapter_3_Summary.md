# Chapter 3: Problem Definition - Summary Document

## ✅ Chapter 3 Complete!

**File**: `Chapter_3_Problem_Definition.md`  
**Size**: ~28 KB  
**Word Count**: ~6,000 words  
**Sections**: 6 major sections with detailed subsections  

---

## 📋 Content Verification

### **Required Section 1: Formal Problem Statement** ✅ INCLUDED

**Location**: Section 3.2 (6 subsections)

**Coverage**:

#### **3.2.1 Basic Code Summarization Problem**
- Traditional formulation: $\text{Summarize}: \mathcal{F} \rightarrow \mathcal{S}$
- Limitations identified

#### **3.2.2 Enhanced Problem Formulation**
- Graph-augmented repository-aware formulation
- $\text{Summarize}_{\text{GAR}}: \mathcal{F} \times \mathcal{P}(R) \times \mathcal{G} \rightarrow \mathcal{S}$
- Quality criteria: Structural accuracy, dependency completeness, factual consistency, NL quality

#### **3.2.3 Multi-View Structural Representation**
- Mathematical definitions for:
  - $G_{\text{AST}}$: Abstract Syntax Tree
  - $G_{\text{CFG}}$: Control Flow Graph
  - $G_{\text{PDG}}$: Program Dependence Graph
  - $G_{\text{CG}}$: Call Graph
- Structural prompt construction: $P(f_i, \mathcal{G}(f_i))$

#### **3.2.4 Repository Context Extraction**
- Relevant context set: $C(f_i, R)$
- Subgraph extraction optimization problem
- Relevance scoring function with proximity, complexity, and CF importance

#### **3.2.5 Agentic Refinement Problem**
- Iterative refinement sequence: $s_0, s_1, \ldots, s_t$
- Decision function for CONSULT/REFINE/FINISH
- Quality threshold $\theta$

#### **3.2.6 Optimization Objective**
- Overall loss function: $\mathcal{L}(\theta) = \mathcal{L}_{\text{gen}} + \lambda_1 \mathcal{L}_{\text{struct}} + \lambda_2 \mathcal{L}_{\text{dep}}$
- Generation, structural, and dependency losses

---

### **Required Section 2: Challenges** ✅ INCLUDED

**Location**: Section 3.5 (3 subsections with 10 challenges)

**Coverage**:

#### **Technical Challenges (5 challenges)**

1. **Multi-View Graph Integration**
   - Problem: Integrate 4 graph types without overwhelming context window
   - Why existing fail: GNNs (opaque), GraphCodeBERT (one type), CodeT5 (none)
   - Our approach: Serialize to text, prioritize critical elements

2. **Scalable Repository-Wide Context Extraction**
   - Problem: Extract relevant context from large repos within limits
   - Why existing fail: Function-level only, semantic retrieval, spatial proximity
   - Our approach: Global call graph, intelligent subgraph extraction

3. **Hallucination Mitigation**
   - Problem: Prevent LLMs from generating incorrect summaries
   - Why existing fail: Single-pass, no verification
   - Our approach: Agentic critique + repository consultation

4. **Explicit Dependency Extraction**
   - Problem: Reliably extract and present "Called by"/"Calls"
   - Why existing fail: No explicit dependency generation
   - Our approach: Call graph + structured prompts + critique verification

5. **Balancing Interpretability and Performance**
   - Problem: Maintain interpretability while achieving competitive performance
   - Why existing fail: GNNs (opaque), pre-trained (implicit)
   - Our approach: Explicit prompts + LoRA fine-tuning

#### **Methodological Challenges (3 challenges)**

6. **Evaluation Metrics for Dependency-Rich Summaries**
   - Problem: Evaluate dependency correctness beyond BLEU/ROUGE
   - Our approach: Dependency coverage metric, structural accuracy metric

7. **Dataset Construction**
   - Problem: Create training data with dependency-rich summaries
   - Our approach: Curated 386 examples + CodeXGlue augmentation

8. **Reproducibility and Fair Comparison**
   - Problem: Ensure reproducible results given methodological issues
   - Our approach: Full documentation, standardized metrics, open-source code

#### **Practical Challenges (2 challenges)**

9. **Computational Efficiency**
   - Problem: Make system practical given graph construction + LLM overhead
   - Our approach: One-time graph construction, early stopping, caching

10. **Generalization to Unseen Code Patterns**
    - Problem: Ensure generalization beyond training data
    - Our approach: Diverse data, structural grounding, out-of-domain evaluation

---

## 📐 Mathematical Formulations

### **Key Equations**

1. **Enhanced Summarization Function**:
   $$\text{Summarize}_{\text{GAR}}: \mathcal{F} \times \mathcal{P}(R) \times \mathcal{G} \rightarrow \mathcal{S}$$

2. **Structural Graph Ensemble**:
   $$\mathcal{G}(f_i) = \{G_{\text{AST}}, G_{\text{CFG}}, G_{\text{PDG}}, G_{\text{CG}}\}$$

3. **Relevant Context Set**:
   $$C(f_i, R) = \text{Callees}(f_i) \cup \text{Callers}(f_i) \cup \text{TransitiveDeps}(f_i, k)$$

4. **Relevance Scoring**:
   $$\text{Relevance}(f_j, f_i) = \alpha \cdot \text{Proximity}(f_j, f_i) + \beta \cdot \text{Complexity}(f_j) + \gamma \cdot \text{CFImportance}(f_j, f_i)$$

5. **Optimization Objective**:
   $$\mathcal{L}(\theta) = \sum_{i=1}^N \left[ \mathcal{L}_{\text{gen}}(s_i, s_i^*) + \lambda_1 \mathcal{L}_{\text{struct}}(s_i, \mathcal{G}(f_i)) + \lambda_2 \mathcal{L}_{\text{dep}}(s_i, C(f_i, R_i)) \right]$$

---

## 📊 Assumptions (10 Total)

### **Code Assumptions (A1-A4)**
- ✅ A1: Well-formed code
- ✅ A2: Reasonable naming conventions
- ✅ A3: Standard repository structure
- ✅ A4: Static analyzability

### **Resource Assumptions (A5-A6)**
- ✅ A5: Computational resources (GPU recommended)
- ✅ A6: Memory availability (≥8GB RAM)

### **Data Assumptions (A7-A8)**
- ✅ A7: Training data quality
- ✅ A8: Ground truth availability

### **Model Assumptions (A9-A10)**
- ✅ A9: LLM instruction following
- ✅ A10: Context window sufficiency

---

## 🎯 Scope and Constraints

### **In Scope**
- ✅ Python code
- ✅ Function/method-level summarization
- ✅ AST, CFG, PDG, Call Graph extraction
- ✅ Repository-wide analysis
- ✅ Dependency-rich summaries
- ✅ Automated + human evaluation

### **Out of Scope**
- ❌ Multi-language support
- ❌ Dynamic analysis
- ❌ Production features (API, cloud, multi-user)
- ❌ Large-scale optimization (>10K files)
- ❌ Interactive features

### **Constraints (C1-C6)**
1. **C1**: Context window ≤ 4,096 tokens
2. **C2**: Repository size ≤ 1,000 files (optimized)
3. **C3**: Inference time ≤ 10 seconds per function
4. **C4**: Model size = Gemma-2b (2B parameters)
5. **C5**: Training data = 386 custom + CodeXGlue
6. **C6**: Human evaluation = sample-based

---

## 🎯 Chapter Structure

```
Chapter 3: Problem Definition

3.1 Introduction
3.2 Formal Problem Statement
    ├─ 3.2.1 Basic Code Summarization
    ├─ 3.2.2 Enhanced Formulation
    ├─ 3.2.3 Multi-View Structural Representation
    ├─ 3.2.4 Repository Context Extraction
    ├─ 3.2.5 Agentic Refinement Problem
    └─ 3.2.6 Optimization Objective
3.3 Assumptions (A1-A10)
    ├─ 3.3.1 Code Assumptions
    ├─ 3.3.2 Resource Assumptions
    ├─ 3.3.3 Data Assumptions
    └─ 3.3.4 Model Assumptions
3.4 Scope and Constraints
    ├─ 3.4.1 Scope (In/Out)
    └─ 3.4.2 Constraints (C1-C6)
3.5 Challenges
    ├─ 3.5.1 Technical Challenges (C1-C5)
    ├─ 3.5.2 Methodological Challenges (C6-C8)
    └─ 3.5.3 Practical Challenges (C9-C10)
3.6 Summary
```

---

## 💡 Key Contributions

### **1. Rigorous Mathematical Formulation**
- First formal definition of graph-augmented repository-aware code summarization
- Explicit quality criteria (structural accuracy, dependency completeness, factual consistency)
- Optimization objective with multiple loss components

### **2. Comprehensive Challenge Analysis**
- 10 challenges organized by category
- For each challenge:
  - ✅ Problem statement
  - ✅ Difficulty analysis
  - ✅ Why existing approaches fail
  - ✅ Our proposed solution
  - ✅ Open research questions

### **3. Clear Boundaries**
- 10 explicit assumptions
- Clear in-scope / out-of-scope delineation
- 6 concrete constraints

### **4. Multi-View Graph Integration**
- Mathematical definitions for all 4 graph types
- Structural prompt construction formulation
- Subgraph extraction as optimization problem

### **5. Agentic Refinement Formalization**
- Iterative refinement as state sequence
- Decision function for CONSULT/REFINE/FINISH
- Quality threshold-based termination

---

## 🔍 Challenge Highlights

### **Most Critical Challenges**

1. **Multi-View Graph Integration** (C1)
   - Core technical challenge
   - Differentiates from all 2021 approaches

2. **Hallucination Mitigation** (C3)
   - Critical for trustworthiness
   - Addressed by agentic workflow

3. **Explicit Dependency Extraction** (C4)
   - Unique requirement
   - No existing work generates this

4. **Evaluation Metrics** (C6)
   - Methodological gap
   - Need new metrics for dependency coverage

5. **Generalization** (C10)
   - Practical concern
   - Structural prompts help but not guaranteed

---

## 📈 Comparison with Chapter 2

### **Chapter 2 (Related Work)**
- Reviewed 2021 state-of-the-art
- Identified 5 research gaps
- Positioned NeuroGraph-CodeRAG

### **Chapter 3 (Problem Definition)**
- Formalized the problem mathematically
- Defined 10 specific challenges
- Specified assumptions and constraints
- Provided solution approaches

**Together**: Chapters 2 and 3 establish **what's been done** (Chapter 2) and **what needs to be solved** (Chapter 3), setting up the solution description in subsequent chapters.

---

## ✅ Quality Checklist

- ✅ **Formal mathematical formulations** (6 subsections)
- ✅ **Clear assumptions** (10 assumptions, 4 categories)
- ✅ **Explicit scope** (in-scope, out-of-scope, constraints)
- ✅ **Comprehensive challenges** (10 challenges, 3 categories)
- ✅ **Rigorous analysis** for each challenge
- ✅ **Academic rigor** with mathematical notation
- ✅ **Clear structure** with logical flow
- ✅ **Connects to Chapter 2** (references 2021 work)

---

## 🎓 Your Thesis Progress

You now have **three complete chapters**:

1. ✅ **Chapter 1: Introduction** (4,200 words)
   - Motivation, Objectives, Background, Research Questions, Contributions

2. ✅ **Chapter 2: Related Work** (5,500 words)
   - 2021 papers only, comprehensive comparison, gap analysis

3. ✅ **Chapter 3: Problem Definition** (6,000 words)
   - Formal problem statement, assumptions, challenges

**Total: ~15,700 words** of high-quality thesis content! 🎉

---

## 🔜 Next Chapters

**Typical thesis structure**:
- ✅ Chapter 1: Introduction
- ✅ Chapter 2: Related Work
- ✅ Chapter 3: Problem Definition
- ⏭️ **Chapter 4: System Architecture/Design**
- ⏭️ **Chapter 5: Implementation**
- ⏭️ **Chapter 6: Experimental Methodology**
- ⏭️ **Chapter 7: Results and Analysis**
- ⏭️ **Chapter 8: Discussion**
- ⏭️ **Chapter 9: Conclusion and Future Work**

---

**End of Summary Document**
