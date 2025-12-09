# CodeT5 Baseline Comparison - Quick Start

## Overview

CodeT5 is a T5-based seq2seq model pretrained for code tasks. We use it as a baseline to compare against our Gemma model with RL techniques.

**Model**: `Salesforce/codet5-base-multi-sum` (pretrained for code summarization)

## Quick Evaluation (5 minutes)

### Zero-shot Evaluation (Recommended)

```bash
python -m src.scripts.evaluate_codet5 --mode zeroshot
```

This evaluates the pretrained CodeT5 model without any fine-tuning.

**Output**: `codet5_zeroshot_results.json`

---

## Optional: Fine-tuning (10-15 minutes)

If you want to compare fine-tuned CodeT5:

### 1. Train CodeT5

```bash
python -m src.model.train_codet5 --epochs 1 --limit 50
```

### 2. Evaluate Fine-tuned Model

```bash
python -m src.scripts.evaluate_codet5 --mode finetuned
```

**Output**: `codet5_finetuned_results.json`

---

## Generate Comparison Report

After running evaluations (and ensuring Gemma results exist):

```bash
python -m src.scripts.compare_models
```

**Outputs**:
- `model_comparison_report.md` - Detailed comparison
- `model_comparison_summary.json` - JSON summary

---

## Expected Results

1. **CodeT5 Zero-shot**: Low scores (not adapted to your dataset)
2. **CodeT5 Fine-tuned**: Moderate scores (minimal training)
3. **Gemma Normal**: Good scores (full training + structural analysis)
4. **Gemma with RL**: **Best scores** (RL-based refinement)

---

## Why CodeT5?

- ✅ **Seq2seq model**: Can actually generate summaries (unlike GraphCodeBERT)
- ✅ **Code-specific**: Pretrained on code tasks
- ✅ **Fair baseline**: Represents state-of-the-art pretrained approach
- ✅ **Fast evaluation**: Zero-shot takes ~5 minutes

---

## Troubleshooting

**Issue**: Model download is slow
- **Solution**: CodeT5-base is ~250MB, first download may take time

**Issue**: Out of memory
- **Solution**: Reduce `--limit` to 25 examples for training

**Issue**: "Fine-tuned model not found"
- **Solution**: Run training first or use `--mode zeroshot`

---

*CodeT5 provides a proper baseline for demonstrating the superiority of your Gemma + RL approach!*
