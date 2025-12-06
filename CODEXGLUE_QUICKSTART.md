# CodeXGlue Integration - Quick Reference

## 🚀 Automated Pipeline (Recommended)

**Single command to do everything:**

```bash
# Download 10K examples, preprocess, split, build RAG, and train (5 epochs)
python run_codexglue_pipeline.py --subset 10000 --epochs 5

# Or use default settings (10K examples, 5 epochs)
python run_codexglue_pipeline.py

# For full dataset (400K+ examples)
python run_codexglue_pipeline.py --full --epochs 10

# Resume from existing files (skip download/preprocess)
python run_codexglue_pipeline.py --skip-download --skip-preprocess --epochs 5
```

**What it does:**
1. ✅ Downloads CodeXGlue dataset
2. ✅ Preprocesses with structural features
3. ✅ Creates train/val/test splits
4. ✅ Builds RAG index
5. ✅ Trains the model

**Time estimate** (10K examples, 4 workers):
- Download: ~2-3 minutes
- Preprocessing: ~10-15 minutes
- Splits: ~1 minute
- RAG index: ~2-3 minutes
- Training (5 epochs): ~2-3 hours (GPU dependent)

---

## 📋 Manual Step-by-Step (Alternative)

## ✅ What's Been Implemented

### Core Scripts (3 new files)
1. **`src/scripts/download_codexglue.py`** - Downloads CodeXGlue dataset from HuggingFace
2. **`src/scripts/preprocess_codexglue.py`** - Deep preprocessing with structural features
3. **`src/scripts/create_dataset_splits.py`** - Stratified train/val/test splitting

### Integration (2 modified files)
1. **`src/data/dataset.py`** - Added multi-dataset support (backward compatible)
2. **`src/model/trainer.py`** - Added dataset selection parameter

### Documentation (3 new files)
1. **`CODEXGLUE_INTEGRATION.md`** - Comprehensive integration guide
2. **`verify_preprocessing.py`** - Preprocessing validation script
3. **`verify_splits.py`** - Split validation script

### Updated Files
1. **`README.md`** - Added dataset selection section

---

## 🚀 Quick Start (Testing)

### Step 1: Download Small Subset (100 examples)
```bash
python -m src.scripts.download_codexglue --subset 100 --output test_codexglue_raw.jsonl --validate
```

### Step 2: Preprocess
```bash
python -m src.scripts.preprocess_codexglue --input test_codexglue_raw.jsonl --output test_codexglue_processed.jsonl --workers 2
```

### Step 3: Validate Preprocessing
```bash
python verify_preprocessing.py test_codexglue_processed.jsonl
```

### Step 4: Create Splits
```bash
python -m src.scripts.create_dataset_splits --input test_codexglue_processed.jsonl --output-dir .
```

### Step 5: Validate Splits
```bash
python verify_splits.py codexglue_train.jsonl codexglue_validation.jsonl codexglue_test.jsonl
```

### Step 6: Test Training (Optional - requires GPU)
```bash
python -m src.model.trainer --dataset-name codexglue --num-train-epochs 1 --output-dir test_model
```

---

## 📊 For Production (10K+ examples)

```bash
# Download 10K subset
python -m src.scripts.download_codexglue --subset 10000 --output codexglue_raw.jsonl --validate

# Preprocess with 8 workers
python -m src.scripts.preprocess_codexglue --input codexglue_raw.jsonl --output codexglue_processed.jsonl --workers 8

# Create splits
python -m src.scripts.create_dataset_splits --input codexglue_processed.jsonl

# Train
python -m src.model.trainer --dataset-name codexglue --num-train-epochs 5 --output-dir gemma_codexglue_finetuned
```

---

## 📝 Key Features

- ✅ **Backward Compatible**: Default behavior unchanged (uses custom dataset)
- ✅ **Multiprocessing**: Parallel preprocessing for speed
- ✅ **Quality Filtering**: Removes invalid/low-quality examples
- ✅ **Stratified Splits**: Balanced complexity distribution
- ✅ **Leakage Detection**: Prevents data contamination
- ✅ **Comprehensive Logging**: Progress tracking and statistics
- ✅ **Validation Scripts**: Automated quality checks

---

## 🔍 What Still Needs Testing

1. **Download script** - Test with small subset first
2. **Preprocessing** - Verify structural features are extracted correctly
3. **Splits** - Confirm no data leakage
4. **Training** - Run on small scale to verify integration
5. **Comparison** - Compare results with custom dataset

---

## 📖 Full Documentation

- **Integration Guide**: `CODEXGLUE_INTEGRATION.md`
- **Implementation Details**: `.gemini/antigravity/brain/.../walkthrough.md`
- **Implementation Plan**: `.gemini/antigravity/brain/.../implementation_plan.md`

---

## ⚠️ Important Notes

- Start with **small subset** (100-1000 examples) for testing
- Preprocessing is **computationally expensive** - use multiprocessing
- Full dataset (400K+) requires **significant time and resources**
- Always **validate** preprocessing and splits before training
