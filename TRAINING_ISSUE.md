# Training Issue: Gemma Model Access Required

## ❌ Current Error

```
Access to model google/gemma-2b-it is restricted. You must have access to it and be authenticated to access it.
```

## ✅ Solution

The Gemma model requires explicit access approval from Google/HuggingFace.

### Step 1: Request Access to Gemma Model

1. Go to: https://huggingface.co/google/gemma-2b-it
2. Click **"Request Access"** or **"Agree and Access"**
3. Accept the terms and conditions
4. Wait for approval (usually instant)

### Step 2: Verify Your HF Token

Make sure you're using a **write token** (not read-only):
1. Go to: https://huggingface.co/settings/tokens
2. Create a new token with **write** permissions if needed
3. Copy the token

### Step 3: Run Training

Once access is granted:

```powershell
# Set your token
$env:HF_TOKEN="your_write_token_here"

# Run training (skipping completed steps)
python run_codexglue_pipeline.py --subset 100 --epochs 1 --skip-download --skip-preprocess --skip-rag
```

---

## 🔄 Alternative: Use a Different Model

If you can't access Gemma, you can modify the code to use an open model like CodeBERT or CodeT5:

### Option A: CodeT5 (No restrictions)

Edit `src/model/model_loader.py` and change:
```python
model_name = "Salesforce/codet5-base"  # Instead of google/gemma-2b-it
```

### Option B: CodeBERT (No restrictions)

```python
model_name = "microsoft/codebert-base"
```

---

## 📊 Current Status

**Completed:**
- ✅ Dataset downloaded (100 examples)
- ✅ Preprocessing complete
- ✅ Splits created (train/val/test)
- ✅ RAG index built

**Blocked:**
- ⏸️ Training (waiting for Gemma access)

**Ready to train as soon as access is granted!**
