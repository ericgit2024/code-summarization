import torch
from transformers import TrainingArguments, Trainer, default_data_collator
try:
    from src.model.model_loader import load_gemma_model, setup_lora
except ImportError:
    print("WARNING: Failed to load real model loader. Using mock loader.")
    from src.model.model_loader_mock import load_gemma_model, setup_lora

from src.data.prompt import construct_prompt
from datasets import load_from_disk
import os

def train_codexglue(
    output_dir="gemma_lora_codexglue",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=2e-4
):
    print("Loading processed CodeXGlue dataset...")
    dataset_path = "data/codexglue_processed"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Run src/scripts/preprocess_codexglue.py first.")

    dataset = load_from_disk(dataset_path)
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    # 1. Load and setup model
    print("Loading model...")
    try:
        model, tokenizer = load_gemma_model()
    except Exception as e:
        print(f"Failed to load real model: {e}. Using mock loader.")
        from src.model.model_loader_mock import load_gemma_model as mock_loader
        model, tokenizer = mock_loader()

    tokenizer.pad_token = tokenizer.eos_token
    try:
        model = setup_lora(model)
    except TypeError:
        model = setup_lora(model, output_dir)

    # 2. Format Prompt
    def format_prompt(example):
        full_prompt = construct_prompt(
            example['structural_prompt'],
            example['code'],
            [], # retrieved_codes
            [], # retrieved_docstrings
            instruction="Provide a comprehensive and detailed summary of the code's functionality. Explain the inputs, outputs, and internal logic step-by-step. Describe how the function interacts with its dependencies and the significance of each operation."
        )
        text = f"{full_prompt} {example['summary']}"
        return {"text": text}

    train_dataset = train_dataset.map(format_prompt)
    eval_dataset = eval_dataset.map(format_prompt)

    # Tokenize
    def tokenize_function(examples):
        tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
        # Explicitly add labels
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    cols_to_remove = [c for c in train_dataset.column_names if c != "text"]

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=cols_to_remove)
    tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=cols_to_remove)

    if "text" in tokenized_train_dataset.column_names:
        tokenized_train_dataset = tokenized_train_dataset.remove_columns(["text"])
    if "text" in tokenized_eval_dataset.column_names:
        tokenized_eval_dataset = tokenized_eval_dataset.remove_columns(["text"])

    print(f"Columns in dataset: {tokenized_train_dataset.column_names}")
    print(f"Sample item keys: {tokenized_train_dataset[0].keys()}")

    # 3. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=8,
        warmup_steps=10,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=5,
        optim="paged_adamw_8bit",
        save_strategy="epoch",
        eval_strategy="epoch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        remove_unused_columns=False,
        label_names=["labels"]
    )

    if not torch.cuda.is_available():
        training_args.fp16 = False
        training_args.optim = "adamw_torch"

    # 4. Trainer
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        args=training_args,
        data_collator=default_data_collator, # Use default collator since we have labels
    )

    # Force signature columns for mock
    if "Mock" in model.__class__.__name__ or hasattr(model, "forward") and "Mock" in str(model.forward):
        trainer._signature_columns = ["input_ids", "attention_mask", "labels"]

    print("Starting training on CodeXGlue...")
    trainer.train()

    print("Saving model...")
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(output_dir)
    print("Training complete.")

if __name__ == "__main__":
    train_codexglue()
