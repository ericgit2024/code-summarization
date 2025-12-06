"""
CodeXGlue Dataset Download Script

Downloads the CodeXGlue code summarization dataset from Hugging Face,
filters for Python language, and converts to JSONL format compatible
with the existing pipeline.

Usage:
    python -m src.scripts.download_codexglue --subset 10000 --output codexglue_raw.jsonl
    python -m src.scripts.download_codexglue --full --output codexglue_raw.jsonl
"""

import argparse
import json
import logging
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_codexglue(subset_size=None, output_file="codexglue_raw.jsonl", language="python"):
    """
    Download CodeXGlue code summarization dataset and convert to JSONL.
    
    Args:
        subset_size (int, optional): Number of examples to download. If None, downloads all.
        output_file (str): Output JSONL file path
        language (str): Programming language to filter (default: "python")
    
    Returns:
        int: Number of examples downloaded
    """
    logger.info(f"Starting CodeXGlue dataset download (language: {language})")
    
    try:
        # Load CodeXGlue code summarization dataset
        # Using the new non-script dataset format (HF removed support for dataset scripts)
        logger.info("Loading dataset from Hugging Face...")
        
        dataset = None
        errors = []
        
        # Method 1: Try CodeXGlue code-to-text (official, non-script version)
        try:
            logger.info("Attempting to load from 'code_x_glue_ct_code_to_text'...")
            dataset = load_dataset(
                "code_x_glue_ct_code_to_text",
                language,
                split="train"
            )
            logger.info("✓ Successfully loaded from code_x_glue_ct_code_to_text")
        except Exception as e1:
            errors.append(f"Method 1 failed: {e1}")
            logger.warning(f"Failed to load from code_x_glue_ct_code_to_text: {e1}")
            
            # Method 2: Try CodeSearchNet v2 (non-script version)
            try:
                logger.info("Attempting to load from 'code_search_net_v2'...")
                dataset = load_dataset(
                    "code_search_net_v2",
                    language,
                    split="train"
                )
                logger.info("✓ Successfully loaded from code_search_net_v2")
            except Exception as e2:
                errors.append(f"Method 2 failed: {e2}")
                logger.warning(f"Failed to load from code_search_net_v2: {e2}")
                
                # Method 3: Try Python-specific community dataset as fallback
                try:
                    logger.info("Attempting to load from 'Nan-Do/code-search-net-python'...")
                    dataset = load_dataset(
                        "Nan-Do/code-search-net-python",
                        split="train"
                    )
                    logger.info("✓ Successfully loaded from Nan-Do/code-search-net-python")
                except Exception as e3:
                    errors.append(f"Method 3 failed: {e3}")
                    logger.error("All dataset loading methods failed!")
                    logger.error("Errors encountered:")
                    for err in errors:
                        logger.error(f"  - {err}")
                    raise RuntimeError(
                        "Could not load CodeSearchNet/CodeXGlue dataset. "
                        "Please check your internet connection and HuggingFace dataset availability. "
                        f"Errors: {'; '.join(errors)}"
                    )
        
        if dataset is None:
            raise RuntimeError("Dataset loading failed - dataset is None")
        
        logger.info(f"Loaded {len(dataset)} total examples")
        
        # Apply subset if specified
        if subset_size is not None and subset_size < len(dataset):
            logger.info(f"Selecting subset of {subset_size} examples")
            dataset = dataset.select(range(subset_size))
        
        # Convert to JSONL format
        logger.info(f"Converting to JSONL format: {output_file}")
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        converted_count = 0
        skipped_count = 0
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in tqdm(dataset, desc="Converting examples"):
                try:
                    # Extract relevant fields from CodeSearchNet format
                    # CodeSearchNet schema varies by source, check available fields
                    
                    # Try different field names for code
                    code = (example.get('whole_func_string') or 
                           example.get('func_code_string') or 
                           example.get('code') or 
                           example.get('function') or '')
                    
                    # Try different field names for summary/documentation
                    summary = (example.get('func_documentation_string') or 
                              example.get('docstring') or 
                              example.get('summary') or 
                              example.get('doc') or '')
                    
                    # Skip examples with empty code or summary
                    if not code.strip() or not summary.strip():
                        skipped_count += 1
                        continue
                    
                    # Try different field names for function name
                    func_name = (example.get('func_name') or 
                                example.get('name') or 
                                example.get('identifier') or '')
                    
                    # Create output record
                    record = {
                        'code': code,
                        'summary': summary,
                        'name': func_name,
                        'language': language,
                        'url': example.get('url', ''),
                        'repo': example.get('repo', ''),
                        'path': example.get('path', '')
                    }
                    
                    # Write as JSON line
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    converted_count += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing example: {e}")
                    skipped_count += 1
                    continue
        
        logger.info(f"✓ Successfully converted {converted_count} examples")
        logger.info(f"✗ Skipped {skipped_count} examples (empty code/summary)")
        logger.info(f"Output saved to: {output_path.absolute()}")
        
        return converted_count
        
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise


def validate_output(output_file):
    """
    Validate the downloaded JSONL file.
    
    Args:
        output_file (str): Path to JSONL file to validate
    
    Returns:
        dict: Validation statistics
    """
    logger.info(f"Validating output file: {output_file}")
    
    stats = {
        'total_examples': 0,
        'has_code': 0,
        'has_summary': 0,
        'has_name': 0,
        'avg_code_length': 0,
        'avg_summary_length': 0,
        'min_code_length': float('inf'),
        'max_code_length': 0,
        'min_summary_length': float('inf'),
        'max_summary_length': 0
    }
    
    code_lengths = []
    summary_lengths = []
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                stats['total_examples'] += 1
                
                if 'code' in record and record['code']:
                    stats['has_code'] += 1
                    code_len = len(record['code'])
                    code_lengths.append(code_len)
                    stats['min_code_length'] = min(stats['min_code_length'], code_len)
                    stats['max_code_length'] = max(stats['max_code_length'], code_len)
                
                if 'summary' in record and record['summary']:
                    stats['has_summary'] += 1
                    summary_len = len(record['summary'])
                    summary_lengths.append(summary_len)
                    stats['min_summary_length'] = min(stats['min_summary_length'], summary_len)
                    stats['max_summary_length'] = max(stats['max_summary_length'], summary_len)
                
                if 'name' in record and record['name']:
                    stats['has_name'] += 1
        
        if code_lengths:
            stats['avg_code_length'] = sum(code_lengths) / len(code_lengths)
        if summary_lengths:
            stats['avg_summary_length'] = sum(summary_lengths) / len(summary_lengths)
        
        # Print validation report
        logger.info("\n" + "="*60)
        logger.info("VALIDATION REPORT")
        logger.info("="*60)
        logger.info(f"Total examples: {stats['total_examples']}")
        
        if stats['total_examples'] > 0:
            logger.info(f"Examples with code: {stats['has_code']} ({stats['has_code']/stats['total_examples']*100:.1f}%)")
            logger.info(f"Examples with summary: {stats['has_summary']} ({stats['has_summary']/stats['total_examples']*100:.1f}%)")
            logger.info(f"Examples with name: {stats['has_name']} ({stats['has_name']/stats['total_examples']*100:.1f}%)")
            logger.info(f"\nCode length stats:")
            logger.info(f"  Average: {stats['avg_code_length']:.1f} chars")
            logger.info(f"  Min: {stats['min_code_length']} chars")
            logger.info(f"  Max: {stats['max_code_length']} chars")
            logger.info(f"\nSummary length stats:")
            logger.info(f"  Average: {stats['avg_summary_length']:.1f} chars")
            logger.info(f"  Min: {stats['min_summary_length']} chars")
            logger.info(f"  Max: {stats['max_summary_length']} chars")
        else:
            logger.warning("No examples found in the dataset!")
        
        logger.info("="*60 + "\n")
        
        return stats
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Download CodeXGlue code summarization dataset"
    )
    parser.add_argument(
        '--subset',
        type=int,
        default=None,
        help='Number of examples to download (default: all)'
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Download full dataset (overrides --subset)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='codexglue_raw.jsonl',
        help='Output JSONL file path (default: codexglue_raw.jsonl)'
    )
    parser.add_argument(
        '--language',
        type=str,
        default='python',
        help='Programming language (default: python)'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate output file after download'
    )
    
    args = parser.parse_args()
    
    # Determine subset size
    subset_size = None if args.full else args.subset
    
    # Download dataset
    try:
        count = download_codexglue(
            subset_size=subset_size,
            output_file=args.output,
            language=args.language
        )
        
        # Validate if requested
        if args.validate:
            validate_output(args.output)
        
        logger.info(f"\n✓ Download complete! {count} examples saved to {args.output}")
        
    except Exception as e:
        logger.error(f"\n✗ Download failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
