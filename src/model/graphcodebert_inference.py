"""
GraphCodeBERT Inference Pipeline

Generates code summaries using GraphCodeBERT (baseline model).
Supports both zero-shot (pretrained) and fine-tuned modes.
"""

from src.model.graphcodebert_loader import load_graphcodebert
import torch


class GraphCodeBERTInference:
    """Inference pipeline for GraphCodeBERT baseline model."""
    
    def __init__(self, use_finetuned=False):
        """
        Initialize the GraphCodeBERT inference pipeline.
        
        Args:
            use_finetuned: If True, use fine-tuned model; otherwise use pretrained
        """
        self.use_finetuned = use_finetuned
        self.model, self.tokenizer = load_graphcodebert(use_finetuned=use_finetuned)
        self.model.eval()
        
        print(f"GraphCodeBERT Inference Pipeline initialized ({'fine-tuned' if use_finetuned else 'zero-shot'})")
    
    def summarize(self, code, max_length=128):
        """
        Generate a summary for the given code.
        
        Args:
            code: Source code string to summarize
            max_length: Maximum length of generated summary
            
        Returns:
            Generated summary string
        """
        # Construct simple prompt
        prompt = f"Summarize the following code:\n\n{code}\n\nSummary:"
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        # Move to same device as model
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate summary
        with torch.no_grad():
            try:
                # Try seq2seq generation
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=1,  # Greedy decoding for speed
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
                # Decode output
                summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
            except Exception as e:
                # Fallback: If model doesn't support generation, use encoder output
                print(f"Generation failed: {e}")
                print("Using fallback: extracting from input (encoder-only model)")
                
                # For encoder-only models, we can't generate - return a placeholder
                summary = "GraphCodeBERT is an encoder-only model and cannot generate summaries without a decoder head."
        
        # Clean up the summary (remove the prompt if it's echoed)
        if "Summary:" in summary:
            summary = summary.split("Summary:")[-1].strip()
        
        return summary
    
    def summarize_batch(self, codes, max_length=128):
        """
        Generate summaries for a batch of code snippets.
        
        Args:
            codes: List of source code strings
            max_length: Maximum length of generated summaries
            
        Returns:
            List of generated summary strings
        """
        summaries = []
        for code in codes:
            summary = self.summarize(code, max_length=max_length)
            summaries.append(summary)
        return summaries


if __name__ == "__main__":
    # Test inference
    test_code = """
def calculate_average(numbers):
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)
"""
    
    print("Testing GraphCodeBERT Inference (Zero-shot)...")
    pipeline = GraphCodeBERTInference(use_finetuned=False)
    summary = pipeline.summarize(test_code)
    
    print(f"\nCode:\n{test_code}")
    print(f"\nGenerated Summary:\n{summary}")
