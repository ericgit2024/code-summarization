from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def load_codet5_model(model_id="Salesforce/codet5-base"):
    """
    Loads the CodeT5 model for sequence-to-sequence tasks.
    """
    print(f"Loading CodeT5 model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    if torch.cuda.is_available():
        model = model.to("cuda")

    return model, tokenizer
