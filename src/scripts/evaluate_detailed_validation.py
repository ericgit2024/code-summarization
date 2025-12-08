"""
Evaluate model performance on detailed validation set.

This script evaluates the model's generated summaries against detailed reference summaries
that match the model's output style, providing accurate BLEU/ROUGE/METEOR scores.

Usage:
    python -m src.scripts.evaluate_detailed_validation --mode [normal|agent]
"""

import json
import argparse
from src.model.inference import InferencePipeline
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


def evaluate_model(model_dir='gemma_lora_finetuned', use_agent=False, limit=None):
    """
    Evaluate the model on the detailed validation set.
    
    Args:
        model_dir: Path to the trained model directory
        use_agent: Whether to use the LangGraph agent mode
        limit: Optional limit on number of examples to evaluate
    """
    logger.info("Loading validation set...")
    validation_examples = load_validation_set()
    
    if limit:
        validation_examples = validation_examples[:limit]
    
    logger.info(f"Loaded {len(validation_examples)} validation examples")
    
    logger.info("Initializing inference pipeline...")
    pipeline = InferencePipeline(model_dir=model_dir)
    
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
    
    mode_name = "Smart Agent (LangGraph)" if use_agent else "Normal"
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating in {mode_name} mode")
    logger.info(f"{'='*60}\n")
    
    for idx, example in enumerate(validation_examples):
        logger.info(f"\n--- Example {idx + 1}/{len(validation_examples)} ---")
        logger.info(f"Code preview: {example['code'][:100]}...")
        
        try:
            # Generate summary
            if use_agent:
                generated_summary = pipeline.summarize_with_agent(code=example['code'])
            else:
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
    logger.info(f"FINAL RESULTS - {mode_name} Mode")
    logger.info(f"{'='*60}")
    logger.info(f"Evaluated {len(results)}/{len(validation_examples)} examples\n")
    
    avg_scores = {}
    for metric, values in all_scores.items():
        if values:
            avg_scores[metric] = sum(values) / len(values)
            logger.info(f"{metric.upper():12s}: {avg_scores[metric]:.4f}")
    
    # Save detailed results
    output_file = f"detailed_validation_results_{'agent' if use_agent else 'normal'}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'mode': mode_name,
            'total_examples': len(validation_examples),
            'evaluated_examples': len(results),
            'average_scores': avg_scores,
            'individual_results': results
        }, f, indent=2)
    
    logger.info(f"\n✅ Detailed results saved to: {output_file}")
    
    return avg_scores


def compare_modes(model_dir='gemma_lora_finetuned', limit=None):
    """
    Compare Normal mode vs Agent mode performance.
    """
    logger.info("\n" + "="*60)
    logger.info("COMPARING NORMAL MODE VS AGENT MODE")
    logger.info("="*60 + "\n")
    
    # Evaluate both modes
    logger.info("Evaluating Normal mode...")
    normal_scores = evaluate_model(model_dir=model_dir, use_agent=False, limit=limit)
    
    logger.info("\n\nEvaluating Agent mode...")
    agent_scores = evaluate_model(model_dir=model_dir, use_agent=True, limit=limit)
    
    # Print comparison
    logger.info("\n" + "="*60)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*60)
    
    metrics = ['bleu_4', 'rouge1', 'rouge2', 'rougeL', 'meteor', 'bert_f1']
    
    logger.info(f"\n{'Metric':<12} | {'Normal':<10} | {'Agent':<10} | {'Improvement':<12}")
    logger.info("-" * 60)
    
    for metric in metrics:
        normal_val = normal_scores.get(metric, 0)
        agent_val = agent_scores.get(metric, 0)
        improvement = ((agent_val - normal_val) / normal_val * 100) if normal_val > 0 else 0
        
        logger.info(f"{metric.upper():<12} | {normal_val:>10.4f} | {agent_val:>10.4f} | {improvement:>+11.2f}%")
    
    # Save comparison
    with open('mode_comparison_results.json', 'w') as f:
        json.dump({
            'normal_mode': normal_scores,
            'agent_mode': agent_scores,
            'improvements': {
                metric: ((agent_scores.get(metric, 0) - normal_scores.get(metric, 0)) / normal_scores.get(metric, 1) * 100)
                for metric in metrics
            }
        }, f, indent=2)
    
    logger.info("\n✅ Comparison results saved to: mode_comparison_results.json")


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on detailed validation set")
    parser.add_argument('--mode', choices=['normal', 'agent', 'compare'], default='agent',
                       help='Evaluation mode: normal, agent, or compare both')
    parser.add_argument('--model_dir', default='gemma_lora_finetuned',
                       help='Path to trained model directory')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of examples to evaluate (for testing)')
    
    args = parser.parse_args()
    
    if args.mode == 'compare':
        compare_modes(model_dir=args.model_dir, limit=args.limit)
    else:
        use_agent = (args.mode == 'agent')
        evaluate_model(model_dir=args.model_dir, use_agent=use_agent, limit=args.limit)


if __name__ == "__main__":
    main()
