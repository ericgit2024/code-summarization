import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_codet5_model(model_id="Salesforce/codet5-base"):
    """
    Loads the CodeT5 model for Seq2Seq tasks.

    Args:
        model_id (str): The Hugging Face model ID.
                        Default is "Salesforce/codet5-base".
    """
    print(f"Loading CodeT5 model: {model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    except Exception as e:
        raise RuntimeError(f"Failed to load CodeT5 model/tokenizer: {e}") from e

    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.to("cuda")

    return model, tokenizer
