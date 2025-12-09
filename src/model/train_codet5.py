"""
CodeT5 Training Script (Fast Mode)

Trains CodeT5 on a small subset of the dataset for quick baseline comparison.
Uses seq2seq training since CodeT5 is a T5-based model.

Usage:
    python -m src.model.train_codet5 --epochs 1 --limit 50
"""

import argparse
import torch
from transformers import TrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from src.model.codet5_loader import load_codet5_model
from src.data.dataset import load_and_process_dataset
import os


def train_codet5(
    output_dir="codet5_finetuned",
    num_train_epochs=1,
    learning_rate=5e-5,
    limit=50
):
    """
    Train CodeT5 on a small subset of data.
    
    Args:
        output_dir: Directory to save the fine-tuned model
        num_train_epochs: Number of training epochs (default: 1 for speed)
        learning_rate: Learning rate
        limit: Number of training examples to use (default: 50 for speed)
    """
    print("="*60)
    print("CodeT5 Fast Training")
    print("="*60)
    print(f"Training on {limit} examples for {num_train_epochs} epoch(s)")
    print("This should take approximately 10-15 minutes")
    print("="*60)
    
    # Load model and tokenizer
    print("\n1. Loading CodeT5 model...")
    model, tokenizer = load_codet5_model(model_id="Salesforce/codet5-base-multi-sum")
    
    # Load dataset
    print("\n2. Loading custom dataset")
    dataset = load_and_process_dataset(split="train")
    
    # Limit dataset size for fast training
    if limit and len(dataset) > limit:
        dataset = dataset.select(range(limit))
        print(f"   Limited to {limit} examples for fast training")
    
    print(f"   Training set size: {len(dataset)}")
    
    # Format prompts for CodeT5
    def format_prompt(example):
        """Create prompt for CodeT5."""
        # CodeT5 format: "summarize: <code>"
        input_text = f"summarize: {example['code']}"
        target_text = example['summary']
        
        return {
            "input_text": input_text,
            "target_text": target_text
        }
    
    print("\n3. Formatting prompts...")
    dataset = dataset.map(format_prompt)
    
    # Tokenize
    def tokenize_function(examples):
        """Tokenize inputs and targets."""
        model_inputs = tokenizer(
            examples["input_text"],
            max_length=512,
            truncation=True,
            padding="max_length"
        )
        
        # Tokenize targets
        labels = tokenizer(
            examples["target_text"],
            max_length=128,
            truncation=True,
            padding="max_length"
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    print("4. Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    print("\n5. Setting up training configuration...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        warmup_steps=10,
        logging_steps=5,
        save_strategy="epoch",
        fp16=True,
        optim="adamw_torch",
        report_to="none",  # Disable wandb/tensorboard
        predict_with_generate=True,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Trainer
    print("\n6. Initializing trainer...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\n7. Starting training...")
    print("="*60)
    trainer.train()
    
    # Save model
    print("\n8. Saving fine-tuned model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("="*60)
    print(f"✅ Training complete! Model saved to: {output_dir}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Train CodeT5 (fast mode)")
    parser.add_argument('--output_dir', default='codet5_finetuned',
                       help='Output directory for fine-tuned model')
    parser.add_argument('--epochs', type=int, default=1,
                       help='Number of training epochs (default: 1)')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='Learning rate (default: 5e-5)')
    parser.add_argument('--limit', type=int, default=50,
                       help='Number of training examples (default: 50)')
    
    args = parser.parse_args()
    
    train_codet5(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        limit=args.limit
    )


if __name__ == "__main__":
    main()
