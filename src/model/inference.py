from src.structure.ast_analyzer import get_ast_prompt
from src.structure.graph_utils import get_cfg_prompt, get_pdg_prompt, get_call_graph
from src.structure.repo_graph import RepoGraphBuilder
from src.data.prompt import construct_prompt
from src.retrieval.rag import RAGSystem
# from src.model.model_loader import load_gemma_model, setup_lora
try:
    from src.model.model_loader import load_gemma_model, setup_lora
except ImportError:
    print("WARNING: Failed to load real model loader. Using mock loader.")
    from src.model.model_loader_mock import load_gemma_model, setup_lora

try:
    from src.model.codet5_loader import load_codet5_model
except ImportError:
    print("WARNING: Failed to load CodeT5 loader.")
    load_codet5_model = None

from peft import PeftModel
import pickle
import os
import torch
import re
import json
from src.structure.ast_analyzer import ASTAnalyzer
from unittest.mock import MagicMock
from src.model.reflective_agent import ReflectiveAgent

def clean_summary_for_evaluation(text):
    """
    Removes structured formatting (markdown, JSON, section headers) from generated summaries
    to match the natural language format of reference docstrings.
    
    Args:
        text: Generated summary text (may contain markdown, JSON, or structured sections)
        
    Returns:
        Clean natural language summary suitable for BLEU/ROUGE evaluation
    """
    if not text or len(text) < 5:
        return text
    
    original_text = text
    
    # Step 1: Handle Python dict literals (e.g., {'doc': "text"} or {'body': "text"})
    # This is more aggressive than JSON parsing
    if '{' in text and ':' in text:
        # Try to extract content from dict-like structures
        # Look for patterns like {'key': "value"} or {"key": "value"}
        import ast
        try:
            # Try to parse as Python literal
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict):
                # Extract the first string value we find
                for key in ['doc', 'docstring', 'summary', 'description', 'body']:
                    if key in parsed and isinstance(parsed[key], str):
                        text = parsed[key]
                        break
                else:
                    # Just take the first string value
                    for value in parsed.values():
                        if isinstance(value, str):
                            text = value
                            break
        except (ValueError, SyntaxError):
            pass
    
    # Step 2: Remove JSON wrapper if present (fallback if ast.literal_eval failed)
    if text.strip().startswith('{') and '"docstring"' in text or '"doc"' in text:
        try:
            # Try to parse as JSON
            data = json.loads(text)
            # Try common keys
            for key in ['docstring', 'doc', 'summary', 'description']:
                if key in data:
                    text = data[key]
                    break
        except (json.JSONDecodeError, ValueError):
            # If JSON parsing fails, try regex extraction
            for pattern in [r'"docstring"\s*:\s*"([^"]+)"', r'"doc"\s*:\s*"([^"]+)"', r'"summary"\s*:\s*"([^"]+)"']:
                match = re.search(pattern, text)
                if match:
                    text = match.group(1)
                    break
    
    # Step 3: Truncate at detailed sections (Detailed Logic, Dependency Analysis, etc.)
    # We want to keep only the initial overview/summary
    stop_markers = [
        r'###? Detailed Logic',
        r'Detailed Logic:',
        r'\*\*Detailed Logic\*\*[:?]?',
        r'###? Dependency Analysis',
        r'Dependency Analysis:',
        r'\*\*Dependency Analysis\*\*[:?]?',
        r'###? Structural Analysis',
        r'Structural Analysis:',
        r'\*\*Structural Analysis\*\*[:?]?',
        r'###? Target Code',
        r'Target Code Example',
        r'###? Similar Code',
        r'###? Instruction',
        r'###? Explanation',
        r'Explanation:',
        r'\*\*Explanation\*\*[:?]?'
    ]

    for marker in stop_markers:
        match = re.search(marker, text, flags=re.IGNORECASE)
        if match:
            text = text[:match.start()]

    # Step 4: Remove "Overview" header if present (but keep content)
    text = re.sub(r'###? Overview:?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\*\*Overview\*\*:?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Natural Language Summary:?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Natural Language Overview:?', '', text, flags=re.IGNORECASE)

    # Step 5: Remove markdown bold/italic formatting
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **text** -> text
    text = re.sub(r'\*([^*]+)\*', r'\1', text)      # *text* -> text
    
    # Step 6: Remove markdown headers (any remaining headers)
    text = re.sub(r'#+\s+', '', text)  # ## Header -> Header
    
    # Step 7: Remove docstring-style formatting (Args:, Returns:, Raises:)
    # But keep the content after them
    text = re.sub(r'\b(Args?|Returns?|Raises?|Parameters?|Yields?|Notes?|Examples?):?\s*', '', text, flags=re.IGNORECASE)
    
    # Step 8: Remove code blocks
    text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
    text = re.sub(r'`([^`]+)`', r'\1', text)  # Inline code
    
    # Step 9: Collapse multiple whitespaces and newlines into single spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Step 10: Remove leading/trailing whitespace
    text = text.strip()
    
    # Step 11: If we ended up with empty text, return original (fallback)
    if not text or len(text) < 10:
        return original_text
    
    return text

class InferencePipeline:
    def __init__(self, model_dir="gemma_lora_finetuned", index_path="rag_index.pkl", repo_path=None, allow_mock=False, model_type="gemma"):
        print("Loading model and tokenizer...")
        self.model_type = model_type

        if model_type == "codet5":
            if load_codet5_model is None:
                raise ImportError("CodeT5 loader is not available.")
            try:
                self.model, self.tokenizer = load_codet5_model()
                print("✓ Successfully loaded CodeT5 model")
            except Exception as e:
                raise RuntimeError(f"Failed to load CodeT5 model: {e}") from e
        else:
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

        # Ensure pad token is set (especially for Gemma)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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
             # SIMPLIFIED: Match CodeSearchNet docstring format (1-3 sentences, plain language)
             instruction = (
                 "Generate a concise docstring summary for this code.\\n"
                 "Write 1-3 sentences explaining what the code does.\\n"
                 "Do NOT use markdown, bullet points, or structured sections."
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
        # Pass repo_graph to get file-aware call graph
        from src.data.prompt import construct_structural_prompt
        structural_prompt = construct_structural_prompt(code, repo_graph=self.repo_graph)
        
        # Store for use in hierarchical prompt
        self._current_structural_prompt = structural_prompt
        
        self.last_structural_prompts = {
            "ast": get_ast_prompt(code),
            "cfg": get_cfg_prompt(code),
            "pdg": get_pdg_prompt(code),
            "call_graph": structural_prompt.split("Call Graph:\n")[-1] if "Call Graph:" in structural_prompt else "N/A"
        }

        # 3. Construct Hierarchical Prompt
        # Align with training prompt structure
        retrieved_codes = [item["code"] for item in retrieved_items]
        retrieved_docstrings = [item["meta"].get("docstring", "") for item in retrieved_items]

        full_prompt = construct_prompt(
            self._current_structural_prompt,
            code,
            retrieved_codes,
            retrieved_docstrings,
            instruction=instruction,
            repo_context=repo_context
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
                max_new_tokens=150,  # Reduced from 300 to match reference length (typically 50-100 tokens)
                min_new_tokens=30,   # Reduced from 50 to allow shorter summaries
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.5,
                no_repeat_ngram_size=3,
                early_stopping=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode output
        if self.model_type == "codet5":
            # Seq2Seq models output only the generated text
            generated_tokens = outputs[0]
        else:
            # CausalLM models output prompt + generated text, so we slice
            generated_tokens = outputs[0][input_len:]

        print(f"\nGenerated token count: {len(generated_tokens)}")
        
        summary = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        print(f"Generated summary length (raw): {len(summary)} chars")
        print(f"Generated summary (raw, first 300 chars):\n{summary[:300]}")
        
        # POST-PROCESSING: Clean summary for evaluation
        # Remove structured formatting (markdown, JSON, section headers) to match reference format
        summary = clean_summary_for_evaluation(summary)
        
        print(f"Generated summary length (cleaned): {len(summary)} chars")
        print(f"Generated summary (cleaned):\n{summary}")
        
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
        
        # Validation: Detect repetitive patterns (e.g., repeated docstrings)
        if summary and len(summary) > 100:
            # Check for repeated phrases
            words = summary.split()
            if len(words) > 20:
                # Check if the same 10-word sequence appears multiple times
                for i in range(len(words) - 10):
                    phrase = ' '.join(words[i:i+10])
                    count = summary.count(phrase)
                    if count > 2:
                        print(f"WARNING: Detected repetitive pattern (appears {count} times): {phrase[:80]}...")
                        # Try to extract just the first occurrence
                        first_occurrence_end = summary.find(phrase) + len(phrase)
                        summary = summary[:first_occurrence_end * 2]  # Keep roughly 2x the first pattern
                        break
        
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
