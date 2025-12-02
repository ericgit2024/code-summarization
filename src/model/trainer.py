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
    num_train_epochs=5, # Increased for better convergence
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
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

    # Split dataset into training and validation
    dataset_split = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]

    def format_prompt(example):
        structural_prompt = construct_structural_prompt(example['code'])

        retrieved_codes = []
        retrieved_docstrings = []
        if rag_system:
            retrieved_codes, retrieved_docstrings, _ = rag_system.retrieve(example['code'], k=3)

        full_prompt = construct_prompt(
            structural_prompt,
            example['code'],
            retrieved_codes,
            retrieved_docstrings,
            instruction="Summarize the code's functionality concisely. Focus on the main purpose, key operations, and important dependencies. Avoid describing every line; instead, capture the high-level logic."
        )

        # Gemma chat format or simple completion?
        # For base model completion, we append the target summary.
        # Format: Prompt + Summary
        text = f"{full_prompt} {example['summary']}"

        return {"text": text}

    train_dataset = train_dataset.map(format_prompt)
    eval_dataset = eval_dataset.map(format_prompt)

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    # 4. Training Arguments
    # Effective batch size = 1 * 8 = 8.
    # Dataset size ~347 (90% of 386). Steps per epoch = 347/8 = 43.
    # 5 epochs = ~215 steps.
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=8,
        warmup_steps=20, # Increased warmup
        num_train_epochs=num_train_epochs, # Use epochs instead of max_steps
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        optim="paged_adamw_8bit",
        save_strategy="epoch",
        eval_strategy="epoch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
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
