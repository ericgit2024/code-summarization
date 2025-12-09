from src.structure.ast_utils import get_structural_prompt
from src.structure.graph_utils import get_cfg, get_pdg, get_call_graph
from src.structure.repo_graph import RepoGraphBuilder
from src.retrieval.rag import RAGSystem
# from src.model.model_loader import load_gemma_model, setup_lora
try:
    from src.model.model_loader import load_gemma_model, setup_lora
except ImportError:
    print("WARNING: Failed to load real model loader. Using mock loader.")
    from src.model.model_loader_mock import load_gemma_model, setup_lora

from peft import PeftModel
import pickle
import os
import torch
from src.structure.ast_analyzer import ASTAnalyzer
from unittest.mock import MagicMock
from src.model.reflective_agent import ReflectiveAgent

class InferencePipeline:
    def __init__(self, model_dir="gemma_lora_finetuned", index_path="rag_index.pkl", repo_path=None, allow_mock=False):
        print("Loading model and tokenizer...")
        # Load base model - no mock fallback unless explicitly allowed
        try:
            self.model, self.tokenizer = load_gemma_model()
            print("✓ Successfully loaded Gemma model")
        except Exception as e:
            if allow_mock:
                print(f"WARNING: Failed to load real model: {e}. Using mock loader.")
                from src.model.model_loader_mock import load_gemma_model as mock_loader
                self.model, self.tokenizer = mock_loader()
            else:
                raise RuntimeError(
                    f"Failed to load Gemma model: {e}\n"
                    "The smart agent requires the real Gemma model to function correctly.\n"
                    "Please ensure the model is properly installed and accessible."
                ) from e

        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load LoRA adapter if exists, else use base model
        if os.path.exists(model_dir):
            print(f"Loading LoRA adapter from {model_dir}...")
            try:
                self.model = PeftModel.from_pretrained(self.model, model_dir)
                print("✓ Successfully loaded LoRA adapter")
            except Exception as e:
                print(f"Warning: Failed to load LoRA adapter: {e}")
                print("Continuing with base model...")
        else:
            print("LoRA adapter not found. Using base model.")

        if hasattr(self.model, "eval"):
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

        # Initialize Reflective Agent
        self.agent = ReflectiveAgent(self)
        
        # Store last structural prompts for UI display
        self.last_structural_prompts = {}

    def build_repo_graph(self, path):
        """Builds the repository graph from a directory or file."""
        if os.path.isdir(path):
            self.repo_graph.build_from_directory(path)
        elif os.path.isfile(path):
            self.repo_graph.build_from_file(path)
        else:
            print(f"Invalid path for repo graph: {path}")

    def summarize(self, code=None, function_name=None, instruction=None):
        if instruction is None:
             instruction = (
                 "Provide a comprehensive and structured summary of the code's functionality.\n"
                 "The output MUST be organized into the following sections using Markdown headers:\n"
                 "1. **Overview**: A high-level explanation of what the code does.\n"
                 "2. **Detailed Logic**: A step-by-step breakdown of the operations, inputs, and outputs.\n"
                 "3. **Dependency Analysis**: An explanation of how the function interacts with external dependencies (e.g., other functions, classes, or APIs), utilizing the provided 'Dependency Context'. Explicitly mention the source file of the dependencies if available (e.g. 'calls function() from filename.py').\n\n"
                 "Ensure the content is detailed and thorough."
             )

        repo_context = None
        target_meta = {}
        
        # If function_name is provided and exists in graph, use it
        if function_name:
            # Check for direct match
            if function_name in self.repo_graph.graph:
                print(f"Summarizing function '{function_name}' from graph...")
                node_data = self.repo_graph.graph.nodes[function_name]
                code = node_data.get("code", code) # Use graph code if available
                repo_context = self.repo_graph.get_context_text(function_name)
                target_meta = node_data.get("metadata", {})

            # Check for partial matches (e.g. user entered "UserService" but graph has "UserService.method")
            else:
                matches = [n for n in self.repo_graph.graph.nodes if function_name in n or n.endswith(f".{function_name}")]

                if not matches and not code:
                    # function_name provided but not found, and no code provided
                    # List first 10 nodes to help user debug
                    available = list(self.repo_graph.graph.nodes())[:10]
                    msg = f"Function '{function_name}' not found in the repository graph.\nAvailable nodes (subset): {available}"
                    raise ValueError(msg)
                elif matches and not code:
                     # User might have meant a class or partial name.
                     # For now, just pick the first one or warn?
                     # Let's fail but with a helpful message about candidates
                     msg = f"Function '{function_name}' not found, but similar nodes exist: {matches[:5]}. Please be specific."
                     raise ValueError(msg)
                else:
                     # function_name provided but not in graph, but code IS provided.
                     # Treat as transient code, maybe ignore function_name or use it as label.
                     print(f"Warning: Function '{function_name}' not found in graph. Analyzing provided code snippet.")

        if code:
             # Analyze transient code
             try:
                 analyzer = ASTAnalyzer(code)
                 res = analyzer.analyze()
                 # Grab first function if any, or general info
                 if res["functions"]:
                     target_meta = list(res["functions"].values())[0]
             except:
                 pass
        else:
            raise ValueError("Either 'code' or 'function_name' must be provided.")

        return self.generate_from_code(code, target_meta, repo_context, instruction)

    def summarize_with_agent(self, code=None, function_name=None):
        """
        Uses the ReflectiveAgent (LangGraph) to generate a summary.
        """
        repo_context = None
        target_meta = {}
        
        # Logic to resolve function_name/code similar to summarize()
        # For POC, let's reuse the resolution logic or assume function_name is valid if provided.
        # Ideally we should extract the resolution logic.
        
        if function_name and function_name in self.repo_graph.graph:
            node_data = self.repo_graph.graph.nodes[function_name]
            code = node_data.get("code", code)
            repo_context = self.repo_graph.get_context_text(function_name)
            target_meta = node_data.get("metadata", {})
        elif code:
             try:
                 analyzer = ASTAnalyzer(code)
                 res = analyzer.analyze()
                 if res["functions"]:
                     target_meta = list(res["functions"].values())[0]
             except:
                 pass
        else:
             raise ValueError("Function not found or no code provided.")

        return self.agent.run(function_name or "unknown", code, repo_context, target_meta)

    def generate_from_code(self, code, metadata, repo_context, instruction):
        """
        Core generation logic, separated for reuse by Agent.
        """
        # 1. Retrieve Context
        retrieved_items = []
        if self.rag_system:
            try:
                retrieved_codes, retrieved_metadata, _ = self.rag_system.retrieve(code, k=3)
                for rc, rm in zip(retrieved_codes, retrieved_metadata):
                    retrieved_items.append({"code": rc, "meta": rm})
            except Exception as e:
                print(f"Retrieval failed: {e}")

        # 2. Capture Structural Prompts for UI Display
        self.last_structural_prompts = {
            "ast": get_structural_prompt(code),
            "cfg": get_cfg(code),
            "pdg": get_pdg(code),
            "call_graph": get_call_graph(code)
        }

        # 3. Construct Hierarchical Prompt
        full_prompt = self.construct_hierarchical_prompt(
            code,
            metadata,
            repo_context,
            retrieved_items,
            instruction
        )

        return self.generate_response(full_prompt)

    def generate_response(self, prompt):
        """
        Generates a response from the model given a raw prompt.
        Uses the real Gemma model - no mock fallback.
        """
        max_input_length = 6000

        # Tokenize input
        if not hasattr(self.tokenizer, "__call__"):
            raise RuntimeError("Tokenizer is not properly initialized. Cannot generate response.")
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length).to(self.model.device)
        input_len = inputs.input_ids.shape[1]
        
        print(f"\n{'='*60}")
        print(f"DEBUG: generate_response() called")
        print(f"{'='*60}")
        print(f"Input Token Length: {input_len}")
        print(f"Prompt (first 500 chars):\n{prompt[:500]}")
        print(f"Prompt (last 200 chars):\n{prompt[-200:]}")
        
        if input_len >= max_input_length:
            print("WARNING: Prompt was truncated! This may lead to poor results.")

        print(f"\nGenerating with model: {type(self.model).__name__}")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.2,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode and strip the prompt from the output by slicing token IDs
        generated_tokens = outputs[0][input_len:]
        print(f"\nGenerated token count: {len(generated_tokens)}")
        
        summary = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        print(f"Generated summary length: {len(summary)} chars")
        print(f"Generated summary (first 300 chars):\n{summary[:300]}")
        
        if not summary or len(summary) < 10:
            print("WARNING: Generated summary is empty or too short!")
            print(f"Full generated text: '{summary}'")
        
        # Validation: Ensure we didn't just echo the prompt
        if summary and len(summary) > 100:
            # Check if the summary contains large chunks of the prompt
            prompt_lines = prompt.split('\n')
            for line in prompt_lines:
                if len(line) > 50 and line in summary:
                    print(f"WARNING: Detected prompt echo in output. Line: {line[:100]}")
        
        # Validation: Ensure minimum quality
        if not summary or len(summary) < 20:
            error_msg = (
                "Error: Model failed to generate a meaningful summary.\n"
                "This could be due to:\n"
                "1. Model not loaded properly (check if using mock model)\n"
                "2. Prompt format incompatible with the model\n"
                "3. Generation parameters need adjustment\n"
                f"Generated text was: '{summary}'"
            )
            print(f"\n{error_msg}")
            return error_msg
        
        print(f"{'='*60}\n")
        return summary

    def construct_hierarchical_prompt(self, code, metadata, repo_context, retrieved_items, instruction):
        """
        Constructs a structured, hierarchical prompt.
        """
        sections = []

        # 1. Instruction
        sections.append(f"### Instruction\n{instruction}\n")

        # 2. Target Function Info
        sections.append("### Target Code Information")
        if metadata:
            args = ", ".join([f"{a['name']}" for a in metadata.get("args", [])])
            sections.append(f"- **Signature**: def {metadata.get('name', 'unknown')}({args})")

            comp = metadata.get("complexity", {})
            sections.append(f"- **Complexity**: Cyclomatic: {comp.get('cyclomatic', 'N/A')}, LOC: {comp.get('loc', 'N/A')}")

            struct = metadata.get("control_structure", {})
            sections.append(f"- **Structure**: Loops: {struct.get('loops', 0)}, Branches: {struct.get('branches', 0)}")
        else:
            sections.append("- Metadata not available.")

        # 3. Dependency Context (Repository Graph)
        if repo_context and repo_context != "No context found.":
            sections.append("\n### Dependency Context (Call Graph)")
            sections.append("The following functions are relevant dependencies identified in the repository:")
            sections.append(repo_context)

        # 4. Similar Code Patterns (RAG)
        if retrieved_items:
            sections.append("\n### Similar Code Patterns")
            sections.append("The following code snippets share similar logic or structure:")
            for i, item in enumerate(retrieved_items):
                meta = item["meta"]
                sections.append(f"\n**Example {i+1}: {meta.get('name', 'snippet')}**")
                doc = meta.get('docstring')
                if doc:
                    sections.append(f"Docstring: {doc.splitlines()[0]}...")
                sections.append(f"Code:\n```python\n{item['code']}\n```")

        # 5. Code to Summarize
        sections.append("\n### Code to Summarize")
        sections.append(f"```python\n{code}\n```")

        # 6. Response Request
        sections.append("\n### Summary")

        return "\n".join(sections)

if __name__ == "__main__":
    from unittest.mock import MagicMock
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
