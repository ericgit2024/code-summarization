# Quick Training Guide - Small Subset for Feasibility Testing

## 🎯 Goal
Complete training ASAP with a small subset (100 examples, 1 epoch) to verify feasibility.

## ✅ What's Already Done
- ✅ Dataset downloaded (100 examples from CodeXGlue)
- ✅ Preprocessing complete with structural features
- ✅ Train/val/test splits created (80/10/10)
- ✅ RAG index built (`rag_index_codexglue.pkl`)

## 📋 What You Need to Do

### Step 1: Get Gemma Model Access
1. Visit: https://huggingface.co/google/gemma-2b-it
2. Click "Agree and Access" or "Request Access"
3. Accept terms (approval is usually instant)

### Step 2: Set Your HF Token
```powershell
$env:HF_TOKEN="your_huggingface_write_token"
```

### Step 3: Run Training (Quick Test)
```powershell
python run_codexglue_pipeline.py --subset 100 --epochs 1 --skip-download --skip-preprocess --skip-rag
```

**Expected time**: ~10-15 minutes

---

## 📊 Training Configuration (Small Subset)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Dataset size | 100 examples | Quick feasibility test |
| Training examples | 80 | After 80/10/10 split |
| Validation examples | 10 | For eval during training |
| Epochs | 1 | Minimal training for testing |
| Batch size | 1 | Memory efficient |
| Learning rate | 2e-4 | Standard for LoRA |
| LoRA rank | 8 | Efficient fine-tuning |

---

## 🔍 What Will Happen

1. **Skip download** - Uses existing `codexglue_raw.jsonl`
2. **Skip preprocessing** - Uses existing `codexglue_processed.jsonl`
3. **Skip RAG** - Uses existing `rag_index_codexglue.pkl`
4. **Load splits** - Loads train/val from existing files
5. **Load model** - Downloads Gemma-2b-it (if not cached)
6. **Apply LoRA** - Adds trainable adapters
7. **Train** - 1 epoch on 80 examples (~10-15 min)
8. **Save** - Model saved to `gemma_codexglue_finetuned/`

---

## 📁 Output Files

After training completes:
```
gemma_codexglue_finetuned/
├── adapter_config.json
├── adapter_model.bin
├── README.md
└── training_args.bin
```

---

## ✅ Verification Steps

After training:

### 1. Check model exists
```powershell
ls gemma_codexglue_finetuned/
```

### 2. Test inference (optional)
```powershell
python -m streamlit run src/ui/app.py
```

### 3. Run benchmark (optional)
```powershell
python -m src.scripts.benchmark
```

---

## 🚀 If Feasibility Test Passes

Scale up to production:
```powershell
python run_codexglue_pipeline.py --subset 10000 --epochs 5
```

**Time**: ~3-4 hours
- 10K examples
- 5 epochs
- Better model quality

---

## ⚠️ Troubleshooting

**If "Access restricted" error:**
- Request access at https://huggingface.co/google/gemma-2b-it
- Use a **write token** (not read-only)

**If out of memory:**
- Reduce batch size (already at 1)
- Use gradient checkpointing (already enabled)
- Reduce subset further: `--subset 50`

**If training is slow:**
- Ensure GPU is being used
- Check CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`

---

## 📝 Summary

**Current status**: Ready to train, just need Gemma access + HF token

**Command to run**:
```powershell
$env:HF_TOKEN="your_token"
python run_codexglue_pipeline.py --subset 100 --epochs 1 --skip-download --skip-preprocess --skip-rag
```

**Time to completion**: ~10-15 minutes

**Next step after feasibility**: Scale to 10K examples if results look good
