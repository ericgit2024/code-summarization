import torch
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
import os
try:
    if os.environ.get("HF_TOKEN") is None:
        print("Warning: HF_TOKEN not set. Using mock model loader.")
        from src.model.model_loader_mock import load_gemma_model, setup_lora
    else:
        from src.model.model_loader import load_gemma_model, setup_lora
except ImportError:
     from src.model.model_loader import load_gemma_model, setup_lora
from src.data.dataset import load_and_process_dataset
from src.data.prompt import construct_structural_prompt, construct_prompt
from src.retrieval.rag import RAGSystem
import pickle
import os

def train(
    output_dir="gemma_lora_finetuned",
    num_train_epochs=2,  # Reduced to 2 for <1 hour training
    per_device_train_batch_size=2,  # Increased to 2 for faster training
    per_device_eval_batch_size=2,   # Increased to 2
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
    print("Loading and preparing custom dataset")
    dataset = load_and_process_dataset(split="train")

    # Split dataset into training and validation
    dataset_split = dataset.train_test_split(test_size=0.15, seed=42)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Evaluation set size: {len(eval_dataset)}")

    def format_prompt(example):
        structural_prompt = construct_structural_prompt(example['code'])

        # ENABLE RAG during training to provide similar examples for learning
        # Using k=2 (not too many) to keep prompts manageable
        retrieved_codes = []
        retrieved_docstrings = []
        if rag_system:
            try:
                retrieved_codes, retrieved_metadata, _ = rag_system.retrieve(example['code'], k=2)
                # Extract docstrings from metadata
                retrieved_docstrings = [m.get('docstring', '') for m in retrieved_metadata]
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}. Continuing without examples.")
                retrieved_codes = []
                retrieved_docstrings = []

        full_prompt = construct_prompt(
            structural_prompt,
            example['code'],
            retrieved_codes,
            retrieved_docstrings,
            instruction=(
                "Generate a concise docstring summary for this code.\n"
                "Write 1-3 sentences explaining what the code does.\n"
                "Do NOT use markdown, bullet points, or structured sections."
            )
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
        # Reduced max_length from 512 to 384 for faster training
        # Most summaries are <100 tokens, so 384 is sufficient
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=384)

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    # 4. Training Arguments - OPTIMIZED FOR <1 HOUR TRAINING
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,  # Increased from 1 to 2 for faster training
        per_device_eval_batch_size=2,   # Increased from 1 to 2
        gradient_accumulation_steps=2,  # Reduced from 4 to 2 (effective batch size = 2*2=4)
        warmup_steps=5, 
        num_train_epochs=2,  # Reduced from 3 to 2 epochs for <1 hour training
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        optim="paged_adamw_8bit",
        save_strategy="epoch",
        eval_strategy="no",
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
