from src.structure.ast_utils import get_structural_prompt
from src.structure.graph_utils import get_cfg
from src.data.prompt import construct_prompt
from src.retrieval.rag import RAGSystem
from src.model.model_loader import load_gemma_model, setup_lora
from peft import PeftModel
import pickle
import os
import torch

class InferencePipeline:
    def __init__(self, model_dir="gemma_lora_finetuned", index_path="rag_index.pkl"):
        print("Loading model and tokenizer...")
        # Load base model
        self.model, self.tokenizer = load_gemma_model()
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load LoRA adapter if exists, else use base model
        if os.path.exists(model_dir):
            print(f"Loading LoRA adapter from {model_dir}...")
            self.model = PeftModel.from_pretrained(self.model, model_dir)
        else:
            print("LoRA adapter not found. Using base model.")

        self.model.eval()

        print("Loading RAG index...")
        if os.path.exists(index_path):
            with open(index_path, "rb") as f:
                self.rag_system = pickle.load(f)
        else:
            print("RAG index not found. Context retrieval will be disabled.")
            self.rag_system = None

    def summarize(self, code, instruction="Summarize the code, focusing on its logic and dependencies."):
        # 1. Extract Structure
        structural_prompt = get_structural_prompt(code)
        cfg_text = get_cfg(code)
        # Combining structure manually here or use construct_structural_prompt if exported
        full_structure = f"AST:\n{structural_prompt}\n\nCFG:\n{cfg_text}"

        # 2. Retrieve Context
        retrieved_codes = []
        retrieved_docstrings = []
        if self.rag_system:
            retrieved_codes, retrieved_docstrings, _ = self.rag_system.retrieve(code, k=3)

        # 3. Construct Prompt
        full_prompt = construct_prompt(
            full_structure,
            code,
            retrieved_codes,
            retrieved_docstrings,
            instruction=instruction
        )

        # 4. Generate
        inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.2
            )

        # Decode and strip the prompt from the output by slicing token IDs
        # Calculate the length of the input tokens to slice the output
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_len:]
        summary = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        return summary

if __name__ == "__main__":
    pipeline = InferencePipeline()
    sample_code = """
    def factorial(n):
        if n == 0:
            return 1
        else:
            return n * factorial(n-1)
    """
    print("Generating summary...")
    summary = pipeline.summarize(sample_code)
    print("\n--- Summary ---\n")
    print(summary)
