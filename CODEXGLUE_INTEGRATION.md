# CodeXGlue Dataset Integration Guide

## Overview

This guide explains how to integrate and use the CodeXGlue code summarization dataset with the NeuroGraph-CodeRAG system. The CodeXGlue dataset is based on CodeSearchNet and provides a large-scale, diverse collection of Python code-summary pairs.

## Quick Start

### 1. Download the Dataset

Download a subset of the CodeXGlue dataset (recommended for initial testing):

```bash
python -m src.scripts.download_codexglue --subset 10000 --output codexglue_raw.jsonl --validate
```

For the full dataset:

```bash
python -m src.scripts.download_codexglue --full --output codexglue_raw.jsonl
```

**Note**: The full Python dataset contains 400K+ examples and may take significant time to download.

### 2. Preprocess the Dataset

Apply deep preprocessing with structural feature extraction:

```bash
python -m src.scripts.preprocess_codexglue --input codexglue_raw.jsonl --output codexglue_processed.jsonl --workers 4
```

This step:
- Validates Python syntax
- Extracts structural features (AST, CFG, PDG, Call Graph)
- Filters low-quality examples
- Calculates cyclomatic complexity
- Counts dependencies

**Note**: Preprocessing is computationally expensive. Use `--workers` to parallelize (default: CPU count - 1).

### 3. Create Train/Validation/Test Splits

Create stratified splits:

```bash
python -m src.scripts.create_dataset_splits --input codexglue_processed.jsonl --output-dir .
```

This creates three files:
- `codexglue_train.jsonl` (80% of data)
- `codexglue_validation.jsonl` (10% of data)
- `codexglue_test.jsonl` (10% of data)

### 4. Build RAG Index (Optional)

Build a FAISS index for retrieval-augmented generation:

```bash
python -m src.scripts.build_rag_index --dataset-name codexglue
```

This creates `rag_index_codexglue.pkl`.

### 5. Train the Model

Train using the CodeXGlue dataset:

```bash
python -m src.model.trainer --dataset-name codexglue --num-train-epochs 5 --output-dir gemma_codexglue_finetuned
```

## Dataset Statistics

### CodeXGlue (Python subset)
- **Total examples**: ~400,000+ (full dataset)
- **Source**: CodeSearchNet corpus
- **Languages**: Python (can be extended to Java, JavaScript, Go, PHP, Ruby)
- **Quality**: Professional code from GitHub repositories
- **Diversity**: Wide range of complexity levels and coding styles

### Comparison with Custom Dataset

| Metric | Custom Dataset | CodeXGlue (10K subset) |
|--------|----------------|------------------------|
| Total examples | 386 | 10,000 |
| Avg code length | ~150 chars | ~300 chars |
| Avg summary length | ~80 chars | ~60 chars |
| Complexity range | 1-15 | 1-50+ |

## Dataset Format

### Raw Format (after download)

```json
{
  "code": "def example_function(x):\n    return x * 2",
  "summary": "Doubles the input value",
  "name": "example_function",
  "language": "python",
  "url": "https://github.com/...",
  "repo": "owner/repo",
  "path": "path/to/file.py"
}
```

### Processed Format (after preprocessing)

```json
{
  "code": "def example_function(x):\n    return x * 2",
  "summary": "Doubles the input value",
  "name": "example_function",
  "complexity": 1,
  "num_dependencies": 0,
  "language": "python",
  "source_url": "https://github.com/...",
  "code_lines": 2,
  "summary_length": 25
}
```

## Advanced Usage

### Custom Subset Size

Download a specific number of examples:

```bash
python -m src.scripts.download_codexglue --subset 50000 --output codexglue_raw.jsonl
```

### Custom Split Ratios

Create custom train/val/test splits:

```bash
python -m src.scripts.create_dataset_splits \
    --input codexglue_processed.jsonl \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --seed 42
```

### Training with Custom Parameters

```bash
python -m src.model.trainer \
    --dataset-name codexglue \
    --num-train-epochs 3 \
    --per-device-train-batch-size 2 \
    --learning-rate 1e-4 \
    --output-dir gemma_codexglue_custom
```

## Troubleshooting

### Issue: Download fails with "Dataset not found"

**Solution**: Ensure you have internet connection and the `datasets` library is installed:
```bash
pip install datasets
```

### Issue: Preprocessing is very slow

**Solution**: Increase the number of workers:
```bash
python -m src.scripts.preprocess_codexglue --input codexglue_raw.jsonl --output codexglue_processed.jsonl --workers 8
```

### Issue: Out of memory during preprocessing

**Solution**: Process in smaller batches or reduce the number of workers:
```bash
python -m src.scripts.preprocess_codexglue --input codexglue_raw.jsonl --output codexglue_processed.jsonl --workers 2
```

### Issue: Training fails with "File not found"

**Solution**: Ensure you've created the splits:
```bash
python -m src.scripts.create_dataset_splits --input codexglue_processed.jsonl
```

## Performance Expectations

### Preprocessing Time (10K examples)
- Single worker: ~30-45 minutes
- 4 workers: ~10-15 minutes
- 8 workers: ~5-8 minutes

### Training Time (10K examples, 5 epochs)
- GPU (RTX 3060): ~2-3 hours
- GPU (RTX 4080): ~1-1.5 hours
- GPU (A100): ~30-45 minutes

### Expected Metrics (after training on 10K examples)
- BLEU: 0.35-0.45
- ROUGE-L: 0.50-0.60
- METEOR: 0.45-0.55
- Semantic Similarity: 0.75-0.85

## Best Practices

1. **Start Small**: Begin with a 1K-10K subset to validate the pipeline before scaling up
2. **Monitor Quality**: Check preprocessing statistics to ensure data quality
3. **Validate Splits**: Always verify no data leakage between splits
4. **Save Checkpoints**: Use `--checkpoint-interval` during preprocessing
5. **Track Experiments**: Use different output directories for different training runs

## Next Steps

After successful integration:
1. Run benchmark evaluation: `python -m src.scripts.benchmark`
2. Compare with custom dataset results
3. Analyze generated summaries for quality
4. Consider fine-tuning hyperparameters
5. Scale up to larger subsets if results are promising

## Support

For issues or questions, refer to:
- Main README: `README.md`
- Implementation plan: `.gemini/antigravity/brain/.../implementation_plan.md`
- Training configuration: `TRAINING_AND_EVALUATION_CONFIG.md`
