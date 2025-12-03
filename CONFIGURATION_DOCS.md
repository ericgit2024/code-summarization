# Configuration Documentation

This document outlines the training and benchmarking configurations for the **NeuroGraph-CodeRAG** (formerly SP-RAG) code summarization system.

## 1. Training Configuration

The training process is defined in `src/model/trainer.py` and uses the Hugging Face `Trainer` API.

### Model & Hardware
*   **Base Model**: `google/gemma-2b-it` (Instruction-tuned 2B parameter model).
*   **Quantization**: 4-bit NormalFloat (NF4) via `bitsandbytes` (QLoRA) to reduce memory usage.
*   **LoRA Configuration**:
    *   Rank (`r`): 8
    *   Alpha (`lora_alpha`): 32
    *   Dropout (`lora_dropout`): 0.05
    *   Target Modules: `q_proj`, `o_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`, `down_proj`.
    *   Task Type: `CAUSAL_LM`.
*   **Gradient Checkpointing**: Enabled (with `use_reentrant=False` to avoid PyTorch warnings).
*   **Precision**: FP16 (Mixed Precision).

### Dataset & Splitting
*   **Source**: Dataset is loaded via `src/data/dataset.py` (which internally loads `code_summary_dataset.jsonl`).
*   **Splitting**:
    *   The `train` split is loaded first.
    *   It is dynamically split into **Train (90%)** and **Validation (10%)** using `dataset.train_test_split(test_size=0.1)`.

### Training Hyperparameters
*   **Epochs**: 5
*   **Batch Size**: 1 per device (effective batch size = 8 via gradient accumulation).
*   **Gradient Accumulation Steps**: 8
*   **Learning Rate**: 2e-4
*   **Optimizer**: `paged_adamw_8bit`
*   **Warmup Steps**: 20
*   **Evaluation Strategy**: Per epoch (`eval_strategy="epoch"`).
*   **Save Strategy**: Per epoch (`save_strategy="epoch"`).
*   **Max Sequence Length**: 512 tokens (defined in `tokenize_function`).

### Prompt Construction
*   **Format**: Structural Prompt + Code + RAG Context + Instruction.
*   **RAG Retrieval**: Retrieves top 3 similar examples (`k=3`) from `rag_index.pkl`.
*   **Instruction**: *"Provide a comprehensive and detailed summary of the code's functionality. Explain the inputs, outputs, and internal logic step-by-step. Describe how the function interacts with its dependencies and the significance of each operation."*

---

## 2. Benchmark Checking Configuration

Benchmarking is performed by two primary scripts: `src/scripts/benchmark.py` (basic metrics) and `src/scripts/evaluate_system.py` (comprehensive evaluation including SAS).

### Reference Data
*   **Ground Truth Source**: The system uses the `'summary'` key from the dataset as the reference summary.
*   **Fallback**: In `benchmark.py`, if `'docstring'` is available, it is prioritized; otherwise, it falls back to `'summary'`. In `evaluate_system.py`, `'summary'` is explicitly used.

### Metrics & Calculation

The system uses `src/utils/metrics.py` to compute standard NLP metrics and `src/utils/structural_metric.py` for custom structural analysis.

#### 1. Standard NLP Metrics
Calculated using the Hugging Face `evaluate` library and `sentence-transformers`.

*   **BLEU**: Bilingual Evaluation Understudy. Measures n-gram overlap.
*   **ROUGE**: Recall-Oriented Understudy for Gisting Evaluation.
    *   `rouge1`: Unigram overlap.
    *   `rouge2`: Bigram overlap.
    *   `rougeL`: Longest Common Subsequence.
*   **METEOR**: Metric for Evaluation of Translation with Explicit ORdering. Considers synonyms and stemming.
*   **Semantic Similarity**:
    *   **Model**: `sentence-transformers/all-MiniLM-L6-v2`.
    *   **Calculation**: Cosine similarity between the embedding of the generated summary and the reference summary.

#### 2. Structural Accuracy Score (SAS)
A custom metric defined in `src/utils/structural_metric.py` to evaluate how well the summary reflects the code's structure.

*   **Score Range**: 0.0 to 1.0.
*   **Components**:
    *   **Dependency Coverage (60% Weight)**:
        *   Extracts function calls from the code using `ASTAnalyzer`.
        *   Checks if the names of these called functions appear in the generated summary.
        *   Formula: `(Count of mentioned dependencies / Total dependencies) * 0.6`.
    *   **Control Flow Awareness (40% Weight)**:
        *   Detects loops (`for`, `while`) and branches (`if`) in the code using `get_cfg`.
        *   Checks if the summary contains keywords like *'loop', 'iterate', 'check', 'condition', 'if', 'when', 'case'*.
        *   Score is 1.0 (if keywords match) or 0.5 (partial credit), weighted by 0.4.

### Execution
*   **Basic Benchmark**: `python3 -m src.scripts.benchmark` (Runs on a subset of the test/validation set).
*   **Comprehensive Eval**: `python3 -m src.scripts.evaluate_system` (Generates `eval_report.json` with detailed SAS breakdown and simulated human scores).
