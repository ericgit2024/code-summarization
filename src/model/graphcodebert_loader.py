"""
GraphCodeBERT Model Loader

Loads Microsoft's GraphCodeBERT model for code summarization baseline comparison.
Supports both pretrained (zero-shot) and fine-tuned modes.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


def load_graphcodebert(model_path="microsoft/graphcodebert-base", use_finetuned=False):
    """
    Load GraphCodeBERT model and tokenizer.
    
    Args:
        model_path: Path to pretrained model or local fine-tuned model
        use_finetuned: If True, load from local fine-tuned checkpoint
        
    Returns:
        model, tokenizer
    """
    if use_finetuned:
        model_path = "graphcodebert_finetuned"
        print(f"Loading fine-tuned GraphCodeBERT from {model_path}...")
    else:
        print(f"Loading pretrained GraphCodeBERT from {model_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model - GraphCodeBERT is an encoder-only model, but we'll use it for generation
    # by loading it as a seq2seq model or using the base model with a generation head
    try:
        # Try loading as seq2seq first
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    except Exception as e:
        print(f"Could not load as Seq2Seq model: {e}")
        print("Loading as base model instead...")
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded successfully!")
    print(f"Model type: {type(model).__name__}")
    print(f"Number of parameters: {model.num_parameters():,}")
    
    return model, tokenizer


if __name__ == "__main__":
    # Test loading
    model, tokenizer = load_graphcodebert()
    print("\nTokenizer special tokens:")
    print(f"  PAD: {tokenizer.pad_token}")
    print(f"  EOS: {tokenizer.eos_token}")
    print(f"  BOS: {tokenizer.bos_token}")
