import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

HF_TOKEN = "YOUR_HF_TOKEN_HERE" # TODO: Replace with your actual Hugging Face token

def load_gemma_model(model_id="google/gemma-2b"):
    """
    Loads the model in 4-bit quantization.

    Args:
        model_id (str): The Hugging Face model ID.
                        Default is "google/gemma-2b".
                        Ensure you have authenticated with `huggingface-cli login` or provided HF_TOKEN.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN
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
