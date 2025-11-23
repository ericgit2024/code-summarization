import torch
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from src.model.model_loader import load_gemma_model, setup_lora
from src.data.dataset import load_and_process_dataset
from src.data.prompt import construct_structural_prompt, construct_prompt
from src.retrieval.rag import RAGSystem
import pickle
import os

def train(
    output_dir="gemma_lora_finetuned",
    num_train_epochs=1, # Keeping it low for demonstration
    per_device_train_batch_size=2, # Low batch size for memory constrained environments
    learning_rate=2e-4,
    index_path="rag_index.pkl"
):
    # 1. Load and setup model
    print("Loading model...")
    model, tokenizer = load_gemma_model()
    tokenizer.pad_token = tokenizer.eos_token
    model = setup_lora(model)
    model.print_trainable_parameters()

    # 2. Load RAG system
    print("Loading RAG index...")
    if os.path.exists(index_path):
        with open(index_path, "rb") as f:
            rag_system = pickle.load(f)
    else:
        print(f"Warning: Index file {index_path} not found. Proceeding without retrieval (empty context).")
        rag_system = None

    # 3. Prepare dataset
    print("Loading and preparing dataset...")
    dataset = load_and_process_dataset(split="train")
    # Use a subset for quicker training demonstration
    dataset = dataset.select(range(100))

    def format_prompt(example):
        structural_prompt = construct_structural_prompt(example['code'])

        retrieved_codes = []
        retrieved_docstrings = []
        if rag_system:
            retrieved_codes, retrieved_docstrings, _ = rag_system.retrieve(example['code'], k=1)

        full_prompt = construct_prompt(
            structural_prompt,
            example['code'],
            retrieved_codes,
            retrieved_docstrings,
            instruction="Summarize the following code, focusing on internal logic and dependencies."
        )

        # Gemma chat format or simple completion?
        # For base model completion, we append the target summary.
        # Format: Prompt + Summary
        text = f"{full_prompt} {example['docstring']}"

        return {"text": text}

    train_dataset = dataset.map(format_prompt)

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    tokenized_dataset = train_dataset.map(tokenize_function, batched=True)

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=10, # Short training for demo
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=1,
        optim="paged_adamw_8bit",
        save_strategy="no", # Don't save checkpoints to save space
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    model.save_pretrained(output_dir)
    print("Training complete.")

if __name__ == "__main__":
    train()
