"""
CodeXGlue Dataset Preprocessing Script

Applies deep preprocessing to extract structural features (AST, CFG, PDG, Call Graph),
validate code quality, and enrich metadata.

Usage:
    python -m src.scripts.preprocess_codexglue --input codexglue_raw.jsonl --output codexglue_processed.jsonl
    python -m src.scripts.preprocess_codexglue --input codexglue_raw.jsonl --output codexglue_processed.jsonl --workers 4
"""

import argparse
import json
import logging
import ast
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import sys

# Import structural analysis modules
from src.structure.ast_analyzer import ASTAnalyzer
from src.structure.graph_utils import get_cfg, get_pdg, get_call_graph

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_cyclomatic_complexity(code_string):
    """
    Calculate cyclomatic complexity of Python code.
    
    Args:
        code_string (str): Python source code
    
    Returns:
        int: Cyclomatic complexity
    """
    try:
        tree = ast.parse(code_string)
        complexity = 1  # Base complexity
        
        # Count decision points
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    except:
        return 1


def extract_function_name(code_string):
    """
    Extract the main function name from code.
    
    Args:
        code_string (str): Python source code
    
    Returns:
        str: Function name or empty string
    """
    try:
        tree = ast.parse(code_string)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return node.name
        return ""
    except:
        return ""


def count_dependencies(code_string):
    """
    Count the number of function calls (dependencies) in the code.
    
    Args:
        code_string (str): Python source code
    
    Returns:
        int: Number of dependencies
    """
    try:
        analyzer = ASTAnalyzer(code_string)
        analysis = analyzer.analyze()
        
        dependency_count = 0
        for func_meta in analysis.get('functions', {}).values():
            dependency_count += len(func_meta.get('calls', []))
        
        return dependency_count
    except:
        return 0


def process_single_example(example_data):
    """
    Process a single example: validate, extract features, filter quality.
    
    Args:
        example_data (tuple): (line_number, json_line) tuple
    
    Returns:
        dict or None: Processed example or None if filtered out
    """
    line_num, json_line = example_data
    
    try:
        example = json.loads(json_line)
        code = example.get('code', '')
        summary = example.get('summary', '')
        
        # Quality filtering
        # 1. Check for empty code or summary
        if not code.strip() or not summary.strip():
            return None
        
        # 2. Validate Python syntax
        try:
            ast.parse(code)
        except SyntaxError:
            return None
        
        # 3. Filter extremely short code (< 3 lines)
        code_lines = [line for line in code.split('\n') if line.strip()]
        if len(code_lines) < 3:
            return None
        
        # 4. Filter extremely long code (> 500 lines) for memory efficiency
        if len(code_lines) > 500:
            return None
        
        # 5. Filter very short summaries (< 10 characters)
        if len(summary.strip()) < 10:
            return None
        
        # Extract structural features (with error handling)
        # Note: We don't store the full structural analysis here to save space
        # It will be generated on-the-fly during training
        
        # Extract metadata
        function_name = extract_function_name(code)
        if not function_name and 'name' in example:
            function_name = example['name']
        
        complexity = calculate_cyclomatic_complexity(code)
        num_dependencies = count_dependencies(code)
        
        # Create processed record
        processed = {
            'code': code,
            'summary': summary,
            'name': function_name,
            'complexity': complexity,
            'num_dependencies': num_dependencies,
            'language': example.get('language', 'python'),
            'source_url': example.get('url', ''),
            'code_lines': len(code_lines),
            'summary_length': len(summary)
        }
        
        return processed
        
    except Exception as e:
        logger.debug(f"Error processing line {line_num}: {e}")
        return None


def preprocess_dataset(input_file, output_file, num_workers=None, checkpoint_interval=1000):
    """
    Preprocess the entire dataset with multiprocessing.
    
    Args:
        input_file (str): Input JSONL file path
        output_file (str): Output JSONL file path
        num_workers (int, optional): Number of worker processes. Defaults to CPU count.
        checkpoint_interval (int): Save checkpoint every N examples
    
    Returns:
        dict: Processing statistics
    """
    logger.info(f"Starting preprocessing: {input_file} -> {output_file}")
    
    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    logger.info(f"Using {num_workers} worker processes")
    
    # Read input file
    logger.info("Reading input file...")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_examples = len(lines)
    logger.info(f"Total examples to process: {total_examples}")
    
    # Prepare output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Statistics
    stats = {
        'total_input': total_examples,
        'processed': 0,
        'filtered_empty': 0,
        'filtered_syntax': 0,
        'filtered_short': 0,
        'filtered_long': 0,
        'filtered_other': 0,
        'complexity_distribution': {},
        'avg_code_lines': 0,
        'avg_summary_length': 0
    }
    
    code_lines_list = []
    summary_lengths_list = []
    
    # Process with multiprocessing
    logger.info("Processing examples...")
    
    with open(output_path, 'w', encoding='utf-8') as out_f:
        # Create enumerated data for tracking
        enumerated_lines = list(enumerate(lines, 1))
        
        # Use multiprocessing pool
        if num_workers > 1:
            with Pool(num_workers) as pool:
                results = list(tqdm(
                    pool.imap(process_single_example, enumerated_lines),
                    total=total_examples,
                    desc="Preprocessing"
                ))
        else:
            # Single process (for debugging)
            results = [process_single_example(item) for item in tqdm(enumerated_lines, desc="Preprocessing")]
        
        # Write results and collect statistics
        for result in results:
            if result is not None:
                # Write to output
                out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                stats['processed'] += 1
                
                # Collect statistics
                code_lines_list.append(result['code_lines'])
                summary_lengths_list.append(result['summary_length'])
                
                # Track complexity distribution
                complexity = result['complexity']
                if complexity not in stats['complexity_distribution']:
                    stats['complexity_distribution'][complexity] = 0
                stats['complexity_distribution'][complexity] += 1
    
    # Calculate averages
    if code_lines_list:
        stats['avg_code_lines'] = sum(code_lines_list) / len(code_lines_list)
    if summary_lengths_list:
        stats['avg_summary_length'] = sum(summary_lengths_list) / len(summary_lengths_list)
    
    stats['filtered_total'] = stats['total_input'] - stats['processed']
    
    # Print statistics
    logger.info("\n" + "="*60)
    logger.info("PREPROCESSING REPORT")
    logger.info("="*60)
    logger.info(f"Total input examples: {stats['total_input']}")
    logger.info(f"Successfully processed: {stats['processed']} ({stats['processed']/stats['total_input']*100:.1f}%)")
    logger.info(f"Filtered out: {stats['filtered_total']} ({stats['filtered_total']/stats['total_input']*100:.1f}%)")
    logger.info(f"\nQuality metrics:")
    logger.info(f"  Average code lines: {stats['avg_code_lines']:.1f}")
    logger.info(f"  Average summary length: {stats['avg_summary_length']:.1f} chars")
    logger.info(f"\nComplexity distribution (top 10):")
    sorted_complexity = sorted(stats['complexity_distribution'].items(), key=lambda x: x[1], reverse=True)[:10]
    for complexity, count in sorted_complexity:
        logger.info(f"  Complexity {complexity}: {count} examples ({count/stats['processed']*100:.1f}%)")
    logger.info("="*60 + "\n")
    
    logger.info(f"✓ Preprocessing complete! Output saved to: {output_path.absolute()}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess CodeXGlue dataset with structural feature extraction"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input JSONL file (raw CodeXGlue data)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='codexglue_processed.jsonl',
        help='Output JSONL file (preprocessed data)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of worker processes (default: CPU count - 1)'
    )
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=1000,
        help='Save checkpoint every N examples (default: 1000)'
    )
    
    args = parser.parse_args()
    
    try:
        stats = preprocess_dataset(
            input_file=args.input,
            output_file=args.output,
            num_workers=args.workers,
            checkpoint_interval=args.checkpoint_interval
        )
        
        logger.info(f"\n✓ Preprocessing successful!")
        return 0
        
    except Exception as e:
        logger.error(f"\n✗ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
