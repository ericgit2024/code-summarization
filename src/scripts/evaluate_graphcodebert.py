"""
Evaluate GraphCodeBERT Baseline Model

Evaluates GraphCodeBERT in both zero-shot and fine-tuned modes on the validation set.
Computes BLEU, ROUGE, METEOR, and BERTScore metrics.

Usage:
    python -m src.scripts.evaluate_graphcodebert --mode both
    python -m src.scripts.evaluate_graphcodebert --mode zeroshot
    python -m src.scripts.evaluate_graphcodebert --mode finetuned
"""

import json
import argparse
import os
from src.model.graphcodebert_inference import GraphCodeBERTInference
from src.evaluation.metrics import CodeSummaryEvaluator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_validation_set(file_path='detailed_validation_set.jsonl'):
    """Load the detailed validation set."""
    examples = []
    with open(file_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def evaluate_graphcodebert(use_finetuned=False, limit=None):
    """
    Evaluate GraphCodeBERT on the validation set.
    
    Args:
        use_finetuned: If True, use fine-tuned model; otherwise use zero-shot
        limit: Optional limit on number of examples to evaluate
        
    Returns:
        Dictionary of average scores
    """
    mode_name = "Fine-tuned" if use_finetuned else "Zero-shot"
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating GraphCodeBERT - {mode_name} Mode")
    logger.info(f"{'='*60}\n")
    
    # Check if fine-tuned model exists
    if use_finetuned and not os.path.exists("graphcodebert_finetuned"):
        logger.error("Fine-tuned model not found! Please train the model first or use --mode zeroshot")
        return None
    
    # Load validation set
    logger.info("Loading validation set...")
    validation_examples = load_validation_set()
    
    if limit:
        validation_examples = validation_examples[:limit]
    
    logger.info(f"Loaded {len(validation_examples)} validation examples")
    
    # Initialize inference pipeline
    logger.info(f"Initializing GraphCodeBERT inference pipeline ({mode_name})...")
    try:
        pipeline = GraphCodeBERTInference(use_finetuned=use_finetuned)
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        return None
    
    # Initialize evaluator
    logger.info("Initializing evaluator...")
    evaluator = CodeSummaryEvaluator()
    
    results = []
    all_scores = {
        'bleu_4': [],
        'rouge1': [],
        'rouge2': [],
        'rougeL': [],
        'meteor': [],
        'bert_f1': []
    }
    
    # Evaluate each example
    for idx, example in enumerate(validation_examples):
        logger.info(f"\n--- Example {idx + 1}/{len(validation_examples)} ---")
        logger.info(f"Code preview: {example['code'][:100]}...")
        
        try:
            # Generate summary
            generated_summary = pipeline.summarize(code=example['code'])
            
            logger.info(f"\nGenerated summary (first 150 chars):\n{generated_summary[:150]}...")
            logger.info(f"\nReference summary (first 150 chars):\n{example['summary'][:150]}...")
            
            # Evaluate
            scores = evaluator.evaluate_summary(example['summary'], generated_summary)
            
            logger.info(f"\nScores:")
            logger.info(f"  BLEU-4:  {scores['bleu_4']:.4f}")
            logger.info(f"  ROUGE-1: {scores['rouge1']:.4f}")
            logger.info(f"  ROUGE-L: {scores['rougeL']:.4f}")
            logger.info(f"  METEOR:  {scores['meteor']:.4f}")
            
            # Store results
            results.append({
                'code': example['code'],
                'reference': example['summary'],
                'generated': generated_summary,
                'scores': scores
            })
            
            # Accumulate scores
            for metric in all_scores:
                all_scores[metric].append(scores[metric])
                
        except Exception as e:
            logger.error(f"Error processing example {idx + 1}: {e}")
            continue
    
    # Calculate averages
    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL RESULTS - GraphCodeBERT {mode_name} Mode")
    logger.info(f"{'='*60}")
    logger.info(f"Evaluated {len(results)}/{len(validation_examples)} examples\n")
    
    avg_scores = {}
    for metric, values in all_scores.items():
        if values:
            avg_scores[metric] = sum(values) / len(values)
            logger.info(f"{metric.upper():12s}: {avg_scores[metric]:.4f}")
    
    # Save detailed results
    output_file = f"graphcodebert_{'finetuned' if use_finetuned else 'zeroshot'}_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'model': f'GraphCodeBERT ({mode_name})',
            'total_examples': len(validation_examples),
            'evaluated_examples': len(results),
            'average_scores': avg_scores,
            'individual_results': results
        }, f, indent=2)
    
    logger.info(f"\n✅ Detailed results saved to: {output_file}")
    
    return avg_scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate GraphCodeBERT baseline model")
    parser.add_argument('--mode', choices=['zeroshot', 'finetuned', 'both'], default='both',
                       help='Evaluation mode: zeroshot, finetuned, or both')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of examples to evaluate (for testing)')
    
    args = parser.parse_args()
    
    if args.mode == 'both':
        logger.info("\n" + "="*60)
        logger.info("EVALUATING BOTH MODES")
        logger.info("="*60 + "\n")
        
        # Evaluate zero-shot first
        logger.info("Step 1/2: Evaluating zero-shot mode...")
        zeroshot_scores = evaluate_graphcodebert(use_finetuned=False, limit=args.limit)
        
        # Evaluate fine-tuned if available
        logger.info("\n\nStep 2/2: Evaluating fine-tuned mode...")
        finetuned_scores = evaluate_graphcodebert(use_finetuned=True, limit=args.limit)
        
        # Print comparison
        if zeroshot_scores and finetuned_scores:
            logger.info("\n" + "="*60)
            logger.info("COMPARISON: Zero-shot vs Fine-tuned")
            logger.info("="*60)
            
            metrics = ['bleu_4', 'rouge1', 'rouge2', 'rougeL', 'meteor', 'bert_f1']
            logger.info(f"\n{'Metric':<12} | {'Zero-shot':<10} | {'Fine-tuned':<10} | {'Improvement':<12}")
            logger.info("-" * 60)
            
            for metric in metrics:
                zs_val = zeroshot_scores.get(metric, 0)
                ft_val = finetuned_scores.get(metric, 0)
                improvement = ((ft_val - zs_val) / zs_val * 100) if zs_val > 0 else 0
                
                logger.info(f"{metric.upper():<12} | {zs_val:>10.4f} | {ft_val:>10.4f} | {improvement:>+11.2f}%")
    
    elif args.mode == 'zeroshot':
        evaluate_graphcodebert(use_finetuned=False, limit=args.limit)
    
    else:  # finetuned
        evaluate_graphcodebert(use_finetuned=True, limit=args.limit)


if __name__ == "__main__":
    main()
