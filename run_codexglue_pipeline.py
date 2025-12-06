"""
End-to-End CodeXGlue Pipeline

This script automates the entire pipeline:
1. Download CodeXGlue dataset
2. Preprocess with structural features
3. Create train/validation/test splits
4. Build RAG index
5. Train the model

Usage:
    python run_codexglue_pipeline.py --subset 10000 --epochs 5
    python run_codexglue_pipeline.py --full --epochs 10
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pipeline(subset_size=None, num_epochs=5, num_workers=None, skip_download=False, skip_preprocess=False, skip_rag=False):
    """
    Run the complete CodeXGlue integration pipeline.
    
    Args:
        subset_size (int, optional): Number of examples to download. None = full dataset
        num_epochs (int): Number of training epochs
        num_workers (int, optional): Number of preprocessing workers
        skip_download (bool): Skip download step if data already exists
        skip_preprocess (bool): Skip preprocessing if already done
        skip_rag (bool): Skip RAG index building
    """
    
    logger.info("="*80)
    logger.info("CODEXGLUE END-TO-END PIPELINE")
    logger.info("="*80)
    logger.info(f"Configuration:")
    logger.info(f"  Dataset size: {'FULL' if subset_size is None else f'{subset_size} examples'}")
    logger.info(f"  Training epochs: {num_epochs}")
    logger.info(f"  Workers: {num_workers or 'auto'}")
    logger.info("="*80)
    
    # File paths
    raw_file = "codexglue_raw.jsonl"
    processed_file = "codexglue_processed.jsonl"
    train_file = "codexglue_train.jsonl"
    val_file = "codexglue_validation.jsonl"
    test_file = "codexglue_test.jsonl"
    rag_index = "rag_index_codexglue.pkl"
    model_output = "gemma_codexglue_finetuned"
    
    try:
        # ====================================================================
        # STEP 1: Download Dataset
        # ====================================================================
        if not skip_download or not Path(raw_file).exists():
            logger.info("\n" + "="*80)
            logger.info("STEP 1/5: Downloading CodeXGlue Dataset")
            logger.info("="*80)
            
            from src.scripts.download_codexglue import download_codexglue, validate_output
            
            count = download_codexglue(
                subset_size=subset_size,
                output_file=raw_file,
                language="python"
            )
            
            logger.info(f"✓ Downloaded {count} examples to {raw_file}")
            
            # Validate
            stats = validate_output(raw_file)
            logger.info(f"✓ Validation complete: {stats['total_examples']} examples")
        else:
            logger.info(f"\n✓ Skipping download - {raw_file} already exists")
        
        # ====================================================================
        # STEP 2: Preprocess Dataset
        # ====================================================================
        if not skip_preprocess or not Path(processed_file).exists():
            logger.info("\n" + "="*80)
            logger.info("STEP 2/5: Preprocessing Dataset")
            logger.info("="*80)
            
            from src.scripts.preprocess_codexglue import preprocess_dataset
            
            stats = preprocess_dataset(
                input_file=raw_file,
                output_file=processed_file,
                num_workers=num_workers,
                checkpoint_interval=1000
            )
            
            logger.info(f"✓ Preprocessed {stats['processed']} examples")
            logger.info(f"  Average code lines: {stats['avg_code_lines']:.1f}")
            logger.info(f"  Average summary length: {stats['avg_summary_length']:.1f} chars")
        else:
            logger.info(f"\n✓ Skipping preprocessing - {processed_file} already exists")
        
        # ====================================================================
        # STEP 3: Create Train/Validation/Test Splits
        # ====================================================================
        if not all(Path(f).exists() for f in [train_file, val_file, test_file]):
            logger.info("\n" + "="*80)
            logger.info("STEP 3/5: Creating Dataset Splits")
            logger.info("="*80)
            
            from src.scripts.create_dataset_splits import load_dataset, create_splits, check_data_leakage, save_split
            
            # Load preprocessed data
            examples = load_dataset(processed_file)
            
            # Create splits
            train_examples, val_examples, test_examples = create_splits(
                examples,
                train_ratio=0.8,
                val_ratio=0.1,
                test_ratio=0.1,
                seed=42
            )
            
            # Check for leakage
            no_leakage = check_data_leakage(train_examples, val_examples, test_examples)
            if not no_leakage:
                raise ValueError("Data leakage detected! Aborting pipeline.")
            
            # Save splits
            save_split(train_examples, train_file)
            save_split(val_examples, val_file)
            save_split(test_examples, test_file)
            
            logger.info(f"✓ Created splits:")
            logger.info(f"  Train: {len(train_examples)} examples")
            logger.info(f"  Validation: {len(val_examples)} examples")
            logger.info(f"  Test: {len(test_examples)} examples")
        else:
            logger.info(f"\n✓ Skipping splits - files already exist")
        
        # ====================================================================
        # STEP 4: Build RAG Index
        # ====================================================================
        if not skip_rag:
            logger.info("\n" + "="*80)
            logger.info("STEP 4/5: Building RAG Index")
            logger.info("="*80)
            
            from src.retrieval.rag import RAGSystem
            from datasets import load_dataset as hf_load_dataset
            import pickle
            
            # Load training data
            logger.info("Loading training data for RAG index...")
            dataset = hf_load_dataset("json", data_files=train_file, split="train")
            
            # Extract codes and docstrings
            codes = [ex['code'] for ex in dataset]
            docstrings = [ex['summary'] for ex in dataset]
            
            logger.info(f"Building FAISS index from {len(codes)} examples...")
            rag_system = RAGSystem()
            rag_system.build_index(codes, docstrings)
            
            # Save index
            with open(rag_index, 'wb') as f:
                pickle.dump(rag_system, f)
            
            logger.info(f"✓ RAG index saved to {rag_index}")
        else:
            logger.info(f"\n✓ Skipping RAG index building")
        
        # ====================================================================
        # STEP 5: Train Model
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("STEP 5/5: Training Model")
        logger.info("="*80)
        
        from src.model.trainer import train
        
        logger.info(f"Starting training with CodeXGlue dataset...")
        logger.info(f"  Epochs: {num_epochs}")
        logger.info(f"  Output directory: {model_output}")
        logger.info(f"  RAG index: {rag_index if not skip_rag else 'rag_index.pkl (default)'}")
        
        train(
            output_dir=model_output,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            learning_rate=2e-4,
            index_path=rag_index if not skip_rag else "rag_index.pkl",
            dataset_name="codexglue"
        )
        
        logger.info(f"✓ Training complete! Model saved to {model_output}")
        
        # ====================================================================
        # PIPELINE COMPLETE
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("✓ PIPELINE COMPLETE!")
        logger.info("="*80)
        logger.info("\nGenerated files:")
        logger.info(f"  Raw data: {raw_file}")
        logger.info(f"  Processed data: {processed_file}")
        logger.info(f"  Train split: {train_file}")
        logger.info(f"  Validation split: {val_file}")
        logger.info(f"  Test split: {test_file}")
        if not skip_rag:
            logger.info(f"  RAG index: {rag_index}")
        logger.info(f"  Trained model: {model_output}/")
        
        logger.info("\nNext steps:")
        logger.info("  1. Run benchmark: python -m src.scripts.benchmark")
        logger.info("  2. Test inference: python -m streamlit run src/ui/app.py")
        logger.info("  3. Compare with custom dataset results")
        
        return 0
        
    except Exception as e:
        logger.error(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end CodeXGlue pipeline: download, preprocess, split, RAG, train"
    )
    
    # Dataset options
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument(
        '--subset',
        type=int,
        default=10000,
        help='Number of examples to download (default: 10000)'
    )
    dataset_group.add_argument(
        '--full',
        action='store_true',
        help='Download full dataset (400K+ examples, overrides --subset)'
    )
    
    # Training options
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Number of training epochs (default: 5)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of preprocessing workers (default: CPU count - 1)'
    )
    
    # Skip options (for resuming)
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip download if raw file exists'
    )
    parser.add_argument(
        '--skip-preprocess',
        action='store_true',
        help='Skip preprocessing if processed file exists'
    )
    parser.add_argument(
        '--skip-rag',
        action='store_true',
        help='Skip RAG index building'
    )
    
    args = parser.parse_args()
    
    # Determine subset size
    subset_size = None if args.full else args.subset
    
    # Run pipeline
    exit_code = run_pipeline(
        subset_size=subset_size,
        num_epochs=args.epochs,
        num_workers=args.workers,
        skip_download=args.skip_download,
        skip_preprocess=args.skip_preprocess,
        skip_rag=args.skip_rag
    )
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
