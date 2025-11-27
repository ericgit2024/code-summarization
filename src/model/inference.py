from src.structure.ast_utils import get_structural_prompt
from src.structure.graph_utils import get_cfg, get_pdg, get_call_graph
from src.structure.repo_graph import RepoGraphBuilder
from src.data.prompt import construct_prompt
from src.retrieval.rag import RAGSystem
from src.model.model_loader import load_gemma_model, setup_lora
from peft import PeftModel
import pickle
import os
import torch

class InferencePipeline:
    def __init__(self, model_dir="gemma_lora_finetuned", index_path="rag_index.pkl", repo_path=None):
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
            
        self.repo_graph = RepoGraphBuilder()
        if repo_path:
            self.build_repo_graph(repo_path)

    def build_repo_graph(self, path):
        """Builds the repository graph from a directory or file."""
        if os.path.isdir(path):
            self.repo_graph.build_from_directory(path)
        elif os.path.isfile(path):
            self.repo_graph.build_from_file(path)
        else:
            print(f"Invalid path for repo graph: {path}")

    def summarize(self, code=None, function_name=None, instruction="Summarize the code, focusing on its logic and dependencies."):
        repo_context = None
        
        # If function_name is provided and exists in graph, use it
        if function_name and function_name in self.repo_graph.graph:
            print(f"Summarizing function '{function_name}' from graph...")
            node_data = self.repo_graph.graph.nodes[function_name]
            code = node_data.get("code", code) # Use graph code if available
            repo_context = self.repo_graph.get_context_text(function_name)
        elif code is None:
            raise ValueError("Either 'code' or 'function_name' must be provided.")

        # 1. Extract Structure
        structural_prompt = get_structural_prompt(code)
        cfg_text = get_cfg(code)
        pdg_text = get_pdg(code)
        cg_text = get_call_graph(code)
        
        # Combining structure manually here or use construct_structural_prompt if exported
        full_structure = f"AST:\n{structural_prompt}\n\nCFG:\n{cfg_text}\n\nPDG:\n{pdg_text}\n\nCall Graph:\n{cg_text}"

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
            instruction=instruction,
            repo_context=repo_context
        )

        # 4. Generate
        inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.2,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode and strip the prompt from the output by slicing token IDs
        # Calculate the length of the input tokens to slice the output
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_len:]
        summary = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        return summary

if __name__ == "__main__":
    # Example usage
    pipeline = InferencePipeline()
    
    # Create a dummy file for testing graph build
    dummy_code = """
def helper():
    return 42

def main():
    x = helper()
    print(x)
"""
    with open("dummy_repo.py", "w") as f:
        f.write(dummy_code)
        
    pipeline.build_repo_graph("dummy_repo.py")
    
    print("Generating summary for 'main'...")
    summary = pipeline.summarize(function_name="main")
    print("\n--- Summary ---\n")
    print(summary)
    
    # Clean up
    if os.path.exists("dummy_repo.py"):
        os.remove("dummy_repo.py")
