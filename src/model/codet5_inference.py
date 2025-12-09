"""
CodeT5 Inference Pipeline

Generates code summaries using CodeT5 (baseline model).
Supports both zero-shot (pretrained) and fine-tuned modes.
"""

from src.model.codet5_loader import load_codet5_model
import torch


class CodeT5Inference:
    """Inference pipeline for CodeT5 baseline model."""
    
    def __init__(self, use_finetuned=False):
        """
        Initialize the CodeT5 inference pipeline.
        
        Args:
            use_finetuned: If True, use fine-tuned model; otherwise use pretrained
        """
        self.use_finetuned = use_finetuned
        
        if use_finetuned:
            model_id = "codet5_finetuned"
        else:
            model_id = "Salesforce/codet5-base-multi-sum"
        
        self.model, self.tokenizer = load_codet5_model(model_id=model_id)
        self.model.eval()
        
        print(f"CodeT5 Inference Pipeline initialized ({'fine-tuned' if use_finetuned else 'zero-shot'})")
    
    def summarize(self, code, max_length=128):
        """
        Generate a summary for the given code.
        
        Args:
            code: Source code string to summarize
            max_length: Maximum length of generated summary
            
        Returns:
            Generated summary string
        """
        # CodeT5 expects input in the format: "summarize: <code>"
        input_text = f"summarize: {code}"
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        # Move to same device as model
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate summary
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=5,  # Beam search for better quality
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            # Decode output
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
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
    
    print("Testing CodeT5 Inference (Zero-shot)...")
    pipeline = CodeT5Inference(use_finetuned=False)
    summary = pipeline.summarize(test_code)
    
    print(f"\nCode:\n{test_code}")
    print(f"\nGenerated Summary:\n{summary}")
