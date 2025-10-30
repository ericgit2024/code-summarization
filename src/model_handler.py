from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig

def load_model_and_tokenizer(model_name="deepseek-ai/deepseek-coder-1.3b-base"):
    """
    Loads the model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
    elif tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    return model, tokenizer

def get_lora_config():
    """
    Returns the LoRA configuration.
    """
    return LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

def apply_lora(model, lora_config):
    """
    Applies LoRA to the model.
    """
    return get_peft_model(model, lora_config)

def save_lora_model(model, lora_model_dir="lora_model"):
    """
    Saves the LoRA adapter.
    """
    model.save_pretrained(lora_model_dir)

def load_lora_model(base_model_name="deepseek-ai/deepseek-coder-1.3b-base", lora_model_dir="lora_model"):
    """
    Loads the base model and the LoRA adapter.
    """
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    config = PeftConfig.from_pretrained(lora_model_dir)
    loaded_model = PeftModel.from_pretrained(base_model, lora_model_dir)
    return loaded_model

if __name__ == '__main__':
    model, tokenizer = load_model_and_tokenizer()
    lora_config = get_lora_config()
    model = apply_lora(model, lora_config)
    model.print_trainable_parameters()
