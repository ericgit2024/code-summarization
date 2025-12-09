"""
GraphCodeBERT Training Script (Fast Mode)

Trains GraphCodeBERT on a small subset of the dataset for quick baseline comparison.
This is optional - we can also use zero-shot evaluation only.

Usage:
    python -m src.model.train_graphcodebert --epochs 1 --limit 50
"""

import argparse
import torch
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from src.model.graphcodebert_loader import load_graphcodebert
from src.data.dataset import load_and_process_dataset
import os


def train_graphcodebert(
    output_dir="graphcodebert_finetuned",
    num_train_epochs=1,
    learning_rate=2e-4,
    limit=50,
    dataset_name="custom"
):
    """
    Train GraphCodeBERT on a small subset of data.
    
    Args:
        output_dir: Directory to save the fine-tuned model
        num_train_epochs: Number of training epochs (default: 1 for speed)
        learning_rate: Learning rate
        limit: Number of training examples to use (default: 50 for speed)
        dataset_name: Dataset to use ("custom" or "codexglue")
    """
    print("="*60)
    print("GraphCodeBERT Fast Training")
    print("="*60)
    print(f"Training on {limit} examples for {num_train_epochs} epoch(s)")
    print(f"This should take approximately 10-15 minutes")
    print("="*60)
    
    # Load model and tokenizer
    print("\n1. Loading GraphCodeBERT model...")
    model, tokenizer = load_graphcodebert(use_finetuned=False)
    
    # Load dataset
    print(f"\n2. Loading dataset: {dataset_name}")
    dataset = load_and_process_dataset(split="train", dataset_name=dataset_name)
    
    # Limit dataset size for fast training
    if limit and len(dataset) > limit:
        dataset = dataset.select(range(limit))
        print(f"   Limited to {limit} examples for fast training")
    
    print(f"   Training set size: {len(dataset)}")
    
    # Format prompts
    def format_prompt(example):
        """Create simple prompt for GraphCodeBERT."""
        # Simple format: code -> summary
        prompt = f"Summarize the following code:\n\n{example['code']}\n\nSummary:"
        target = example['summary']
        
        return {
            "input_text": prompt,
            "target_text": target
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
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        warmup_steps=10,
        logging_steps=5,
        save_strategy="epoch",
        fp16=True,
        optim="adamw_torch",
        report_to="none",  # Disable wandb/tensorboard
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Trainer
    print("\n6. Initializing trainer...")
    trainer = Trainer(
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
    parser = argparse.ArgumentParser(description="Train GraphCodeBERT (fast mode)")
    parser.add_argument('--output_dir', default='graphcodebert_finetuned',
                       help='Output directory for fine-tuned model')
    parser.add_argument('--epochs', type=int, default=1,
                       help='Number of training epochs (default: 1)')
    parser.add_argument('--lr', type=float, default=2e-4,
                       help='Learning rate (default: 2e-4)')
    parser.add_argument('--limit', type=int, default=50,
                       help='Number of training examples (default: 50)')
    parser.add_argument('--dataset', default='custom',
                       choices=['custom', 'codexglue'],
                       help='Dataset to use')
    
    args = parser.parse_args()
    
    train_graphcodebert(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        limit=args.limit,
        dataset_name=args.dataset
    )


if __name__ == "__main__":
    main()
