"""
GraphCodeBERT Training Script (Fast Mode)

Trains GraphCodeBERT on a small subset of the dataset for quick baseline comparison.
Uses Masked Language Modeling (MLM) since GraphCodeBERT is an encoder-only model.

Usage:
    python -m src.model.train_graphcodebert --epochs 1 --limit 50
"""

import argparse
import torch
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling, RobertaForMaskedLM
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
    Train GraphCodeBERT on a small subset of data using MLM.
    
    Args:
        output_dir: Directory to save the fine-tuned model
        num_train_epochs: Number of training epochs (default: 1 for speed)
        learning_rate: Learning rate
        limit: Number of training examples to use (default: 50 for speed)
        dataset_name: Dataset to use ("custom" or "codexglue")
    """
    print("="*60)
    print("GraphCodeBERT Fast Training (MLM)")
    print("="*60)
    print(f"Training on {limit} examples for {num_train_epochs} epoch(s)")
    print("Using Masked Language Modeling approach")
    print("="*60)
    
    # Load model with MLM head
    print("\n1. Loading GraphCodeBERT model with MLM head...")
    from transformers import RobertaTokenizer
    
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
    model = RobertaForMaskedLM.from_pretrained("microsoft/graphcodebert-base")
    
    print(f"Model loaded successfully!")
    print(f"Model type: {type(model).__name__}")
    print(f"Number of parameters: {model.num_parameters():,}")
    
    # Load dataset
    print(f"\n2. Loading dataset: {dataset_name}")
    dataset = load_and_process_dataset(split="train", dataset_name=dataset_name)
    
    # Limit dataset size for fast training
    if limit and len(dataset) > limit:
        dataset = dataset.select(range(limit))
        print(f"   Limited to {limit} examples for fast training")
    
    print(f"   Training set size: {len(dataset)}")
    
    # Format for MLM - just use code and summary together
    def format_for_mlm(example):
        """Create text for masked language modeling."""
        # Combine code and summary for MLM training
        text = f"Code: {example['code']}\nSummary: {example['summary']}"
        return {"text": text}
    
    print("\n3. Formatting for MLM...")
    dataset = dataset.map(format_for_mlm)
    
    # Tokenize
    def tokenize_function(examples):
        """Tokenize inputs for MLM."""
        return tokenizer(
            examples["text"],
            max_length=512,
            truncation=True,
            padding="max_length"
        )
    
    print("4. Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    print("\n5. Setting up training configuration...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=2,  # Increased since we're not doing seq2seq
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        warmup_steps=10,
        logging_steps=5,
        save_strategy="epoch",
        fp16=True,
        optim="adamw_torch",
        report_to="none",  # Disable wandb/tensorboard
    )
    
    # Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15  # Standard MLM masking probability
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
    print("\nNote: This model was trained with MLM (Masked Language Modeling).")
    print("For inference, you'll need to use it differently than a seq2seq model.")


def main():
    parser = argparse.ArgumentParser(description="Train GraphCodeBERT (fast mode with MLM)")
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
