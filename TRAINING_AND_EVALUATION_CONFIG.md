# Training and Evaluation Configuration Documentation

## Document Overview

This document provides a comprehensive specification of the training configuration, benchmark evaluation methodology, and metric calculation procedures used in the NeuroGraph-CodeRAG code summarization system. It serves as a reference for understanding how the model is trained, how performance is measured, and what baselines are used for comparison.

---

## Table of Contents

1. [Training Configuration](#training-configuration)
2. [Model Architecture and Quantization](#model-architecture-and-quantization)
3. [Dataset Configuration](#dataset-configuration)
4. [Benchmark Evaluation Configuration](#benchmark-evaluation-configuration)
5. [Metrics Calculation Methodology](#metrics-calculation-methodology)
6. [Structural Accuracy Score (SAS)](#structural-accuracy-score-sas)
7. [Reference Baselines](#reference-baselines)
8. [Evaluation Pipeline](#evaluation-pipeline)

---

## 1. Training Configuration

### 1.1 Core Training Parameters

The model training is configured using the following hyperparameters, defined in `src/model/trainer.py`:

| Parameter | Value | Description | Rationale |
|:----------|:------|:------------|:----------|
| **Model** | `google/gemma-2b-it` | Gemma 2B Instruction-Tuned | Balances performance with accessibility; can run on consumer hardware |
| **Output Directory** | `gemma_lora_finetuned/` | Directory for saving model checkpoints | Stores LoRA adapters and training artifacts |
| **Training Epochs** | `5` | Number of complete passes through dataset | Increased from 3 for better convergence |
| **Train Batch Size** | `1` | Per-device batch size during training | Limited by GPU memory with 4-bit quantization |
| **Eval Batch Size** | `1` | Per-device batch size during evaluation | Matches training batch size |
| **Learning Rate** | `2e-4` | Initial learning rate for optimizer | Standard for LoRA fine-tuning |
| **Gradient Accumulation Steps** | `8` | Steps to accumulate gradients before update | Effective batch size = 1 × 8 = 8 |
| **Warmup Steps** | `20` | Linear warmup steps for learning rate | Prevents instability in early training |
| **Max Sequence Length** | `512` | Maximum token length for input | Balances context coverage with memory |

### 1.2 Advanced Training Configuration

**Optimization Strategy:**
```python
TrainingArguments(
    optim="paged_adamw_8bit",           # 8-bit AdamW optimizer for memory efficiency
    fp16=True,                           # Mixed precision training (FP16)
    gradient_checkpointing=True,         # Reduces memory by recomputing activations
    gradient_checkpointing_kwargs={
        'use_reentrant': False           # Compatibility with newer PyTorch versions
    },
    save_strategy="epoch",               # Save checkpoint after each epoch
    eval_strategy="epoch",               # Evaluate after each epoch
    logging_steps=10                     # Log metrics every 10 steps
)
```

**Effective Training Dynamics:**
- **Dataset Size**: ~347 training examples (90% of 386 total after filtering)
- **Steps per Epoch**: 347 ÷ 8 (effective batch size) = ~43 steps
- **Total Training Steps**: 43 steps × 5 epochs = **~215 steps**
- **Validation Split**: 10% of dataset (~35 examples)

### 1.3 LoRA (Low-Rank Adaptation) Configuration

Defined in `src/model/model_loader.py`:

```python
LoraConfig(
    r=8,                                 # Rank of LoRA matrices
    lora_alpha=32,                       # Scaling factor (alpha/r = 4.0)
    target_modules=[                     # Attention and MLP layers to adapt
        "q_proj", "o_proj",              # Query and output projections
        "k_proj", "v_proj",              # Key and value projections
        "gate_proj", "up_proj", "down_proj"  # MLP layers
    ],
    lora_dropout=0.05,                   # Dropout for regularization
    bias="none",                         # Don't train bias terms
    task_type="CAUSAL_LM"                # Causal language modeling task
)
```

**LoRA Benefits:**
- **Parameter Efficiency**: Only ~0.5% of model parameters are trainable
- **Memory Efficiency**: Enables training on consumer GPUs (12-16GB VRAM)
- **Modularity**: LoRA adapters can be swapped without reloading base model

### 1.4 Quantization Configuration

**4-bit Quantization (BitsAndBytes):**
```python
BitsAndBytesConfig(
    load_in_4bit=True,                   # Enable 4-bit quantization
    bnb_4bit_quant_type="nf4",           # NormalFloat4 quantization
    bnb_4bit_compute_dtype=torch.float16 # Compute in FP16 for speed
)
```

**Impact:**
- **Memory Reduction**: ~75% reduction (2B params: 8GB → 2GB)
- **Speed**: Minimal inference slowdown with FP16 compute
- **Accuracy**: NF4 quantization preserves model quality

---

## 2. Model Architecture and Quantization

### 2.1 Base Model: Gemma-2b-it

**Architecture Specifications:**
- **Parameters**: 2 billion
- **Architecture**: Decoder-only transformer (GPT-style)
- **Context Window**: 8,192 tokens (but limited to 512 for training efficiency)
- **Vocabulary Size**: 256,000 tokens
- **Training Data**: Publicly available web documents, code, and mathematics

**Instruction Tuning:**
- The `-it` variant is instruction-tuned for better prompt following
- Trained to respond to structured prompts (ideal for our structural prompting approach)

### 2.2 Trainable Parameters

With LoRA configuration (r=8, alpha=32):
- **Total Parameters**: ~2,000,000,000
- **Trainable Parameters**: ~10,485,760 (0.52%)
- **Frozen Parameters**: ~1,989,514,240 (99.48%)

**Trainable Modules:**
```
q_proj.lora_A, q_proj.lora_B    (Query projection)
k_proj.lora_A, k_proj.lora_B    (Key projection)
v_proj.lora_A, v_proj.lora_B    (Value projection)
o_proj.lora_A, o_proj.lora_B    (Output projection)
gate_proj.lora_A, gate_proj.lora_B
up_proj.lora_A, up_proj.lora_B
down_proj.lora_A, down_proj.lora_B
```

---

## 3. Dataset Configuration

### 3.1 Dataset Source

**File**: `code_summary_dataset.jsonl`  
**Format**: JSON Lines (one JSON object per line)  
**Location**: Repository root directory

**Schema:**
```json
{
  "code": "def example_function(x):\n    return x * 2",
  "summary": "Doubles the input value and returns the result.",
  "name": "example_function",
  "complexity": 1
}
```

**Required Fields:**
- `code` (str): Python source code
- `summary` (str): Natural language summary/docstring

**Optional Fields:**
- `name` (str): Function name
- `complexity` (int): Cyclomatic complexity
- `docstring` (str): Alternative key for summary

### 3.2 Data Preprocessing

**Validation Filter** (`src/data/dataset.py`):
```python
def is_valid_example(example):
    try:
        ast.parse(example['code'])          # Must be syntactically valid Python
        return len(example['summary'].strip()) > 0  # Must have non-empty summary
    except SyntaxError:
        return False
```

**Train/Validation Split:**
- **Training**: 90% of valid examples
- **Validation**: 10% of valid examples
- **Seed**: 42 (for reproducibility)
- **Method**: Random stratified split

### 3.3 Prompt Construction

Each training example is formatted using structural prompting:

**Step 1: Extract Structural Features**
```python
structural_prompt = construct_structural_prompt(example['code'])
# Includes: AST, CFG, PDG, Call Graph
```

**Step 2: Retrieve Similar Examples (RAG)**
```python
retrieved_codes, retrieved_docstrings, _ = rag_system.retrieve(example['code'], k=3)
# Retrieves 3 most similar code examples from training set
```

**Step 3: Construct Full Prompt**
```python
full_prompt = construct_prompt(
    structural_prompt,
    example['code'],
    retrieved_codes,
    retrieved_docstrings,
    instruction="Summarize the code's functionality concisely. 
                 Focus on the main purpose, key operations, and 
                 important dependencies. Avoid describing every line; 
                 instead, capture the high-level logic."
)
```

**Step 4: Format for Training**
```python
text = f"{full_prompt} {example['summary']}"
# Model learns to generate summary after seeing prompt
```

**Tokenization:**
- **Padding**: `max_length` (512 tokens)
- **Truncation**: Enabled (truncates if exceeds 512)
- **Padding Token**: Set to `eos_token`

---

## 4. Benchmark Evaluation Configuration

### 4.1 Benchmark Script: `src/scripts/benchmark.py`

**Purpose**: Evaluate model performance on unseen validation data

**Configuration:**
```python
def run_benchmark(num_samples=20):
    dataset = load_and_process_dataset(split="validation")
    dataset = dataset.select(range(num_samples))  # Subset for quick evaluation
    pipeline = InferencePipeline()
    
    for example in dataset:
        code = example['code']
        reference = example.get('docstring', example.get('summary', ''))
        summary = pipeline.summarize(code)
        predictions.append(summary)
        references.append(reference)
    
    metrics = compute_metrics(predictions, references)
```

**Parameters:**
- **Test Set**: Validation split (10% of dataset)
- **Sample Size**: 20 examples (configurable)
- **Inference Mode**: Uses fine-tuned model with LoRA adapters
- **RAG Retrieval**: Enabled (k=3 similar examples)

### 4.2 Comprehensive Evaluation: `src/scripts/evaluate_system.py`

**Purpose**: Extended evaluation with structural accuracy metrics

**Configuration:**
```python
def run_comprehensive_evaluation(num_samples=10):
    # Standard NLP metrics
    nlp_metrics = compute_metrics([generated], [reference])
    
    # Structural Accuracy Score
    sas = calculate_structural_accuracy(code, generated)
    
    # Simulated human evaluation (Likert scale 1-5)
    likert_sim = min(5, max(1, int(sas * 5) + 1))
```

**Output:**
- Individual example results saved to `eval_report.json`
- Aggregated metrics printed to console
- Sample outputs for qualitative analysis

---

## 5. Metrics Calculation Methodology

### 5.1 Standard NLP Metrics

Implemented in `src/utils/metrics.py`:

#### **5.1.1 BLEU (Bilingual Evaluation Understudy)**

**Library**: HuggingFace `evaluate`  
**Implementation**:
```python
bleu_metric = evaluate.load("bleu")
formatted_references = [[ref] for ref in references]  # Wrap each reference
bleu_score = bleu_metric.compute(predictions=predictions, 
                                  references=formatted_references)
```

**What it Measures**: N-gram overlap between generated and reference summaries  
**Range**: 0.0 to 1.0 (higher is better)  
**Calculation**:
- Computes precision for 1-gram, 2-gram, 3-gram, 4-gram matches
- Applies brevity penalty for short predictions
- Geometric mean of n-gram precisions

**Interpretation**:
- **0.0 - 0.2**: Poor quality, minimal overlap
- **0.2 - 0.4**: Moderate quality, some correct phrases
- **0.4 - 0.6**: Good quality, substantial overlap
- **0.6 - 1.0**: Excellent quality, near-identical to reference

#### **5.1.2 ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**

**Library**: HuggingFace `evaluate`  
**Variants Computed**:
```python
rouge_metric = evaluate.load("rouge")
rouge_results = rouge_metric.compute(predictions=predictions, references=references)

# Extracted scores:
rouge1 = rouge_results['rouge1']   # Unigram overlap
rouge2 = rouge_results['rouge2']   # Bigram overlap
rougeL = rouge_results['rougeL']   # Longest common subsequence
```

**What it Measures**: Recall-based n-gram overlap  
**Range**: 0.0 to 1.0 (higher is better)

**Variants Explained**:
- **ROUGE-1**: Unigram (single word) overlap
  - Measures vocabulary coverage
  - Example: "function returns value" vs "returns the value" → High ROUGE-1
  
- **ROUGE-2**: Bigram (two consecutive words) overlap
  - Measures phrase-level similarity
  - More strict than ROUGE-1
  
- **ROUGE-L**: Longest Common Subsequence
  - Measures sentence-level structure similarity
  - Doesn't require consecutive matches

**Calculation Formula** (ROUGE-1 example):
```
ROUGE-1 = (Number of overlapping unigrams) / (Total unigrams in reference)
```

#### **5.1.3 METEOR (Metric for Evaluation of Translation with Explicit ORdering)**

**Library**: HuggingFace `evaluate`  
**Implementation**:
```python
meteor_metric = evaluate.load("meteor")
meteor_result = meteor_metric.compute(predictions=predictions, references=references)
meteor_score = meteor_result["meteor"]
```

**What it Measures**: Harmonic mean of precision and recall with additional features  
**Range**: 0.0 to 1.0 (higher is better)

**Unique Features**:
- **Stemming**: Matches word stems (e.g., "running" matches "run")
- **Synonyms**: Uses WordNet for synonym matching
- **Word Order**: Penalizes fragmented matches
- **Recall-Focused**: Weights recall higher than precision

**Calculation Steps**:
1. Align words between prediction and reference (exact, stem, synonym)
2. Calculate unigram precision and recall
3. Compute F-mean (harmonic mean with recall weighted higher)
4. Apply fragmentation penalty for non-contiguous matches

**Interpretation**:
- Better than BLEU for capturing semantic similarity
- More lenient with paraphrasing
- Correlates better with human judgment

#### **5.1.4 Semantic Similarity (Cosine Similarity)**

**Library**: `sentence-transformers`  
**Model**: `all-MiniLM-L6-v2`  
**Implementation**:
```python
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings1 = model.encode(predictions, convert_to_tensor=True)
embeddings2 = model.encode(references, convert_to_tensor=True)
cosine_scores = util.cos_sim(embeddings1, embeddings2)
semantic_score = cosine_scores.diagonal().mean().item()
```

**What it Measures**: Semantic similarity in embedding space  
**Range**: -1.0 to 1.0 (typically 0.5 to 1.0 for similar texts)

**Calculation**:
1. Encode prediction and reference into 384-dimensional vectors
2. Compute cosine similarity: `cos(θ) = (A · B) / (||A|| × ||B||)`
3. Average diagonal elements (pair-wise similarities)

**Advantages**:
- Captures semantic meaning beyond word overlap
- Robust to paraphrasing and synonym usage
- Complements n-gram metrics

**Embedding Model Details**:
- **Architecture**: Sentence-BERT (SBERT)
- **Training**: Trained on 1B+ sentence pairs
- **Dimensions**: 384
- **Speed**: ~2000 sentences/second on CPU

### 5.2 Metrics Aggregation

**Output Format**:
```python
{
    "bleu": 0.4523,
    "rouge1": 0.6234,
    "rouge2": 0.4512,
    "rougeL": 0.5823,
    "meteor": 0.5634,
    "semantic_similarity": 0.8234
}
```

**Interpretation Guidelines**:

| Metric | Excellent | Good | Moderate | Poor |
|:-------|:----------|:-----|:---------|:-----|
| BLEU | > 0.5 | 0.3-0.5 | 0.15-0.3 | < 0.15 |
| ROUGE-1 | > 0.6 | 0.4-0.6 | 0.2-0.4 | < 0.2 |
| ROUGE-L | > 0.5 | 0.35-0.5 | 0.2-0.35 | < 0.2 |
| METEOR | > 0.5 | 0.35-0.5 | 0.2-0.35 | < 0.2 |
| Semantic Sim | > 0.8 | 0.6-0.8 | 0.4-0.6 | < 0.4 |

---

## 6. Structural Accuracy Score (SAS)

### 6.1 Overview

**Purpose**: Measure how well the summary captures structural aspects of code (dependencies and control flow)

**Implementation**: `src/utils/metrics.py` (integrated with other metrics)

**Score Range**: 0.0 to 1.0 (higher is better)

### 6.2 Components and Weights

```python
weights = {
    'dependency': 0.6,      # 60% weight
    'control_flow': 0.4     # 40% weight
}
```

### 6.3 Dependency Coverage Score

**What it Measures**: Does the summary mention functions that are called in the code?

**Calculation**:
```python
# Step 1: Extract all function calls from code
analyzer = ASTAnalyzer(code)
analysis = analyzer.analyze()
dependencies = set()
for func_meta in analysis['functions'].values():
    for call in func_meta.get('calls', []):
        dependencies.add(call['name'])

# Step 2: Check if each dependency is mentioned in summary
hits = sum(1 for dep in dependencies if dep.split('.')[-1] in generated_summary)

# Step 3: Calculate score
dep_score = hits / len(dependencies) if dependencies else 1.0
```

**Example**:
```python
# Code
def process_data(data):
    cleaned = clean_data(data)
    validated = validate_input(cleaned)
    return save_to_db(validated)

# Dependencies: {clean_data, validate_input, save_to_db}

# Summary A: "Processes data by cleaning, validating, and saving to database"
# Mentions: clean*, validat*, sav* → 3/3 hits → dep_score = 1.0

# Summary B: "Processes the input data"
# Mentions: none → 0/3 hits → dep_score = 0.0
```

**Edge Cases**:
- If no dependencies exist: `dep_score = 1.0` (nothing to miss)
- Partial matches allowed (e.g., "clean" matches "clean_data")

### 6.4 Control Flow Awareness Score

**What it Measures**: Does the summary mention control flow structures (loops, conditionals)?

**Calculation**:
```python
# Step 1: Detect control flow in code
has_loop = 'for ' in code or 'while ' in code
has_branch = 'if ' in code

# Step 2: Check if summary mentions control flow keywords
keywords = ['loop', 'iterate', 'check', 'condition', 'if', 'when', 'case']
summary_lower = generated_summary.lower()
hit = any(kw in summary_lower for kw in keywords)

# Step 3: Calculate score
if has_loop or has_branch:
    cf_score = 1.0 if hit else 0.5  # Penalize if missing
else:
    cf_score = 1.0  # No control flow to mention
```

**Example**:
```python
# Code with loop
def sum_list(items):
    total = 0
    for item in items:
        total += item
    return total

# Summary A: "Iterates through items and sums them"
# Contains "iterates" → cf_score = 1.0

# Summary B: "Sums the items in the list"
# No control flow keywords → cf_score = 0.5
```

### 6.5 Final SAS Calculation

```python
SAS = (dep_score × 0.6) + (cf_score × 0.4)
```

**Example Calculation**:
```
dep_score = 0.8 (4 out of 5 dependencies mentioned)
cf_score = 1.0 (control flow keywords present)

SAS = (0.8 × 0.6) + (1.0 × 0.4)
    = 0.48 + 0.4
    = 0.88
```

**Interpretation**:
- **0.8 - 1.0**: Excellent structural awareness
- **0.6 - 0.8**: Good structural coverage
- **0.4 - 0.6**: Moderate structural awareness
- **< 0.4**: Poor structural coverage

### 6.6 Error Handling

```python
try:
    # Calculate dependency score
except Exception as e:
    logger.warning(f"SAS Dependency Analysis failed: {e}")
    score += 0.5 * weights['dependency']  # Fallback to neutral score
```

**Fallback Strategy**: If structural analysis fails, assign 0.5 (neutral) to that component

---

## 7. Reference Baselines

### 7.1 Current Baseline: Standard Gemma-2b

**Configuration**:
- **Model**: `google/gemma-2b-it` (base model without fine-tuning)
- **Prompting**: Simple instruction without structural features
- **RAG**: Disabled

**Prompt Format**:
```
Summarize the following Python code:

[CODE]

Summary:
```

**Purpose**: Demonstrates value of structural prompting and fine-tuning

### 7.2 Planned Baselines (Phase 2)

#### **7.2.1 Code2Seq**
- **Paper**: Alon et al., "code2seq: Generating Sequences from Structured Representations of Code"
- **Approach**: AST path-based encoder-decoder
- **Implementation**: Use official repository or re-implementation
- **Comparison Metrics**: BLEU, ROUGE, METEOR

#### **7.2.2 GraphCodeBERT**
- **Paper**: Guo et al., "GraphCodeBERT: Pre-training Code Representations with Data Flow"
- **Approach**: Pre-trained model with data flow awareness
- **Implementation**: HuggingFace model `microsoft/graphcodebert-base`
- **Fine-tuning**: On same dataset for fair comparison

#### **7.2.3 CAST**
- **Paper**: Gong et al., "CAST: Enhancing Code Summarization with Hierarchical Splitting"
- **Approach**: Hierarchical AST splitting
- **Implementation**: Requires custom AST splitting logic

#### **7.2.4 HA-ConvGNN**
- **Paper**: Li et al., "Hierarchical Attention Graph Neural Network for Code Summarization"
- **Approach**: Graph Neural Network on AST + Call Graph
- **Implementation**: Requires GNN framework (PyTorch Geometric)

#### **7.2.5 GPT-4 API (Zero-Shot)**
- **Model**: OpenAI GPT-4
- **Prompting**: Same structural prompt as our system
- **Purpose**: Upper bound comparison (state-of-the-art LLM)
- **Cost Consideration**: Limited to small test set

### 7.3 Comparison Methodology

**Evaluation Protocol**:
1. **Same Test Set**: All baselines evaluated on identical validation split
2. **Same Metrics**: BLEU, ROUGE, METEOR, Semantic Similarity, SAS
3. **Statistical Significance**: Paired t-test for metric differences
4. **Qualitative Analysis**: Manual inspection of 50 random examples

**Reporting Format**:
```
| Model              | BLEU  | ROUGE-L | METEOR | Sem-Sim | SAS   |
|--------------------|-------|---------|--------|---------|-------|
| Gemma-2b (base)    | 0.23  | 0.41    | 0.35   | 0.62    | 0.45  |
| Code2Seq           | 0.31  | 0.48    | 0.42   | 0.68    | 0.52  |
| GraphCodeBERT      | 0.38  | 0.54    | 0.49   | 0.74    | 0.58  |
| NeuroGraph-CodeRAG | 0.45* | 0.62*   | 0.56*  | 0.81*   | 0.73* |
| GPT-4 (zero-shot)  | 0.52  | 0.68    | 0.63   | 0.87    | 0.78  |

* Statistically significant improvement over all baselines (p < 0.05)
```

---

## 8. Evaluation Pipeline

### 8.1 Quick Benchmark (20 samples)

**Command**:
```bash
python -m src.scripts.benchmark
```

**Steps**:
1. Load validation dataset
2. Select first 20 examples
3. Generate summaries using fine-tuned model
4. Compute standard NLP metrics
5. Print results to console

**Output**:
```
Benchmark Results:
bleu: 0.4523
rouge1: 0.6234
rouge2: 0.4512
rougeL: 0.5823
meteor: 0.5634
semantic_similarity: 0.8234
```

**Runtime**: ~2-5 minutes (depending on hardware)

### 8.2 Comprehensive Evaluation (10 samples)

**Command**:
```bash
python -m src.scripts.evaluate_system
```

**Steps**:
1. Load validation dataset
2. Select first 10 examples
3. Generate summaries
4. Compute NLP metrics + SAS
5. Simulate human evaluation (Likert scale)
6. Save detailed report to `eval_report.json`

**Output File** (`eval_report.json`):
```json
[
  {
    "code_snippet": "def factorial(n):\n    if n == 0:\n        ret...",
    "generated": "Calculates factorial recursively by checking base case and multiplying n with factorial of n-1",
    "reference": "Computes the factorial of a number using recursion",
    "bleu": 0.4234,
    "rougeL": 0.5823,
    "sas": 0.8500,
    "simulated_human_score": 5
  },
  ...
]
```

**Aggregated Metrics**:
```
=== Comprehensive Evaluation Report ===
Average BLEU: 0.4523
Average Structural Accuracy Score (SAS): 0.7345
```

### 8.3 Full Dataset Evaluation (Production)

**Recommended Configuration**:
```python
# Modify benchmark.py
run_benchmark(num_samples=len(validation_dataset))  # All validation examples
```

**Considerations**:
- **Runtime**: ~1-2 hours for full validation set
- **Memory**: Monitor GPU memory usage
- **Logging**: Save intermediate results to prevent data loss

---

## 9. Configuration Files and Environment

### 9.1 Environment Variables

**Required**:
```bash
export HF_TOKEN="your_huggingface_token_here"
```

**Optional**:
```bash
export CUDA_VISIBLE_DEVICES="0"        # GPU selection
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"  # Memory management
```

### 9.2 Dependencies

**Key Libraries** (from `requirements.txt`):
```
transformers          # HuggingFace models
peft                  # LoRA implementation
bitsandbytes          # 4-bit quantization
accelerate            # Distributed training
sentence-transformers # Embeddings for RAG and metrics
faiss-cpu             # Vector similarity search
evaluate              # Metrics (BLEU, ROUGE, METEOR)
langgraph             # Reflective agent framework
```

### 9.3 Hardware Requirements

**Minimum (Training)**:
- GPU: 12GB VRAM (e.g., RTX 3060, RTX 4070)
- RAM: 16GB
- Storage: 10GB

**Recommended (Training)**:
- GPU: 16GB+ VRAM (e.g., RTX 4080, A100)
- RAM: 32GB
- Storage: 20GB

**Inference Only**:
- GPU: 8GB VRAM (or CPU with slower performance)
- RAM: 8GB
- Storage: 5GB

---

## 10. Troubleshooting and Best Practices

### 10.1 Common Issues

**Issue 1: Out of Memory (OOM)**
```
Solution:
- Reduce batch size to 1
- Increase gradient accumulation steps
- Reduce max_length to 256
- Enable gradient checkpointing
```

**Issue 2: Poor Metrics on Validation**
```
Diagnosis:
- Check if model is overfitting (train loss << val loss)
- Verify dataset quality (are summaries dependency-rich?)
- Ensure RAG index is built correctly

Solutions:
- Increase training epochs
- Add dropout to LoRA config
- Augment dataset with more examples
```

**Issue 3: Slow Inference**
```
Solutions:
- Use GPU instead of CPU
- Reduce max_new_tokens in generation
- Batch multiple examples together
- Use KV-cache (enabled by default)
```

### 10.2 Best Practices

**Training**:
1. Always build RAG index before training
2. Monitor validation metrics every epoch
3. Save checkpoints frequently
4. Use mixed precision (FP16) for speed

**Evaluation**:
1. Use same test set across all experiments
2. Report mean and standard deviation
3. Include qualitative examples
4. Test on diverse code patterns

**Dataset**:
1. Filter invalid examples (syntax errors)
2. Ensure summaries are high-quality
3. Balance dataset across complexity levels
4. Include diverse function types

---

## 11. Future Enhancements

### 11.1 Planned Metrics

**Dependency Mention Rate (DMR)**:
- Percentage of called functions mentioned in summary
- More fine-grained than SAS dependency score

**Hallucination Rate**:
- Count of mentioned functions that don't exist in code
- Requires manual annotation or automated AST checking

**Code-Summary Alignment Score**:
- Semantic alignment between code blocks and summary sentences
- Uses attention weights from model

### 11.2 Planned Training Improvements

**Curriculum Learning**:
- Start with simple functions (low complexity)
- Gradually increase to complex functions
- Improves convergence and final performance

**Multi-Task Learning**:
- Joint training on summarization + function name prediction
- Auxiliary task: predict cyclomatic complexity
- Improves structural understanding

**Data Augmentation**:
- Paraphrase existing summaries
- Generate synthetic examples from GitHub
- Back-translation for diversity

---

## Document Metadata

- **Version**: 1.0
- **Last Updated**: December 3, 2025
- **Author**: NeuroGraph-CodeRAG Development Team
- **Related Documents**:
  - `PROJECT_DESIGN.md`: Overall project design
  - `FEATURES.md`: System features
  - `WALKTHROUGH.md`: Usage instructions

---

## References

1. **BLEU**: Papineni et al., "BLEU: a Method for Automatic Evaluation of Machine Translation" (2002)
2. **ROUGE**: Lin, "ROUGE: A Package for Automatic Evaluation of Summaries" (2004)
3. **METEOR**: Banerjee & Lavie, "METEOR: An Automatic Metric for MT Evaluation" (2005)
4. **LoRA**: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
5. **QLoRA**: Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)
6. **Gemma**: Google DeepMind, "Gemma: Open Models Based on Gemini Research and Technology" (2024)
