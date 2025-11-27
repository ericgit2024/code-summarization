import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

def load_gemma_model(model_id="google/gemma-2b-it"):
    """
    Loads the model in 4-bit quantization.

    Args:
        model_id (str): The Hugging Face model ID.
                        Default is "google/gemma-2b-it".
                        
    Environment Variables:
        HF_TOKEN: Your Hugging Face authentication token (required for gated models like Gemma)
    """
    # Get HuggingFace token from environment variable
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN environment variable not set. "
            "Please set it with your Hugging Face token: "
            "export HF_TOKEN='your_token_here' (Linux/Mac) or "
            "$env:HF_TOKEN='your_token_here' (Windows PowerShell)"
        )
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token
    )

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    return model, tokenizer

def setup_lora(model):
    """
    Configures and applies LoRA to the model.
    """
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    return model
