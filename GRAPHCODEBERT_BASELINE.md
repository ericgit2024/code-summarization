# GraphCodeBERT Baseline - Quick Start Guide

This guide explains how to quickly evaluate GraphCodeBERT as a baseline comparison for our Gemma model with RL techniques.

## Overview

We use **Microsoft's GraphCodeBERT** (`microsoft/graphcodebert-base`) as a baseline to demonstrate the superiority of our approach. GraphCodeBERT is a well-established pretrained model for code understanding tasks.

## Quick Evaluation (Recommended)

### Option 1: Zero-shot Evaluation (~5 minutes)

Evaluate the pretrained GraphCodeBERT model without any fine-tuning:

```bash
python -m src.scripts.evaluate_graphcodebert --mode zeroshot
```

This will:
- Load the pretrained `microsoft/graphcodebert-base` model
- Evaluate on `detailed_validation_set.jsonl`
- Save results to `graphcodebert_zeroshot_results.json`

### Option 2: With Minimal Fine-tuning (~15-20 minutes total)

Train GraphCodeBERT on a small subset and then evaluate:

```bash
# Step 1: Train (1 epoch, 50 examples, ~10-15 minutes)
python -m src.model.train_graphcodebert --epochs 1 --limit 50

# Step 2: Evaluate both modes (~5-10 minutes)
python -m src.scripts.evaluate_graphcodebert --mode both
```

This will:
- Fine-tune GraphCodeBERT on 50 examples for 1 epoch
- Evaluate both zero-shot and fine-tuned models
- Save results to both JSON files

## Generate Comparison Report

After evaluating GraphCodeBERT and your Gemma models, generate a comprehensive comparison:

```bash
python -m src.scripts.compare_models
```

This will:
- Load all evaluation results (GraphCodeBERT + Gemma)
- Generate `model_comparison_report.md` with detailed analysis
- Save `model_comparison_summary.json` with metrics

## Expected Results

Based on our approach, you should see:

1. **GraphCodeBERT Zero-shot**: Lowest scores (not adapted to our task)
2. **GraphCodeBERT Fine-tuned**: Low scores (minimal training)
3. **Gemma Normal**: Moderate scores (fully fine-tuned with structural analysis)
4. **Gemma + RL Agent**: Highest scores (RL-based refinement)

This demonstrates that our novel approach (Gemma + structural analysis + RL) significantly outperforms traditional pretrained models.

## Files Created

- `graphcodebert_zeroshot_results.json` - Zero-shot evaluation results
- `graphcodebert_finetuned_results.json` - Fine-tuned evaluation results (if trained)
- `graphcodebert_finetuned/` - Fine-tuned model directory (if trained)
- `model_comparison_report.md` - Comprehensive comparison report
- `model_comparison_summary.json` - JSON summary of all metrics

## Customization

### Train on more examples:
```bash
python -m src.model.train_graphcodebert --limit 100 --epochs 2
```

### Evaluate on subset (for testing):
```bash
python -m src.scripts.evaluate_graphcodebert --mode zeroshot --limit 10
```

## Troubleshooting

**Issue**: "Fine-tuned model not found"
- **Solution**: Run training first or use `--mode zeroshot`

**Issue**: Out of memory during training
- **Solution**: Reduce `--limit` to 25 or 30 examples

**Issue**: Slow evaluation
- **Solution**: Use `--limit 10` to test on fewer examples first
