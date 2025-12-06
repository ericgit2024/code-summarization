import os
import ast
from datasets import load_dataset, DatasetDict
from src.data.prompt import construct_structural_prompt

def is_valid_python(code):
    try:
        ast.parse(code)
        return True
    except:
        return False

def preprocess_and_save():
    print("Loading CodeXGlue (Python) dataset...")
    # Load a subset for feasibility in this environment
    # Using 'python' config for CodeXGlue code-to-text
    train_ds = load_dataset("google/code_x_glue_ct_code_to_text", "python", split="train[:200]")
    val_ds = load_dataset("google/code_x_glue_ct_code_to_text", "python", split="validation[:20]")

    print(f"Initial size: {len(train_ds)} train, {len(val_ds)} val.")

    # Filter invalid python code
    print("Filtering invalid Python code...")
    train_ds = train_ds.filter(lambda x: is_valid_python(x['code']))
    val_ds = val_ds.filter(lambda x: is_valid_python(x['code']))
    print(f"Filtered size: {len(train_ds)} train, {len(val_ds)} val.")

    def add_structural_info(example):
        try:
            # Generate structural prompt (AST, CFG, etc.)
            # This is computationally expensive, hence doing it in preprocessing
            struct = construct_structural_prompt(example['code'])
            return {
                "structural_prompt": struct,
                "summary": example['docstring'], # Rename docstring to summary to match existing trainer expectation
                "valid": True
            }
        except Exception as e:
            # Return dummy or mark invalid
            return {
                "structural_prompt": "",
                "summary": example['docstring'],
                "valid": False
            }

    print("Computing structural prompts (this may take a while)...")
    train_ds = train_ds.map(add_structural_info)
    val_ds = val_ds.map(add_structural_info)

    # Filter out any that failed structural analysis
    train_ds = train_ds.filter(lambda x: x['valid'])
    val_ds = val_ds.filter(lambda x: x['valid'])

    print(f"Final size: {len(train_ds)} train, {len(val_ds)} val.")

    output_path = "data/codexglue_processed"
    os.makedirs(output_path, exist_ok=True)

    final_dataset = DatasetDict({
        "train": train_ds,
        "validation": val_ds
    })

    final_dataset.save_to_disk(output_path)
    print(f"Saved processed dataset to {output_path}")

if __name__ == "__main__":
    preprocess_and_save()
