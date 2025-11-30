import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.model.inference import InferencePipeline

def test_poc():
    print("Initializing Pipeline...")
    # Use mock loader if possible to avoid loading heavy model, but we need generation.
    # The inference pipeline handles mock fallback if imports fail, but we want to force mock for speed if we don't have GPU.
    # However, we want to test the AGENT logic.
    
    pipeline = InferencePipeline()
    
    # Create a dummy repo with dependencies
    dummy_code = """
def database_connect():
    print("Connecting to DB...")
    return True

def fetch_users():
    if database_connect():
        return ["Alice", "Bob"]
    return []

def process_users():
    users = fetch_users()
    for user in users:
        print(f"Processing {user}")
"""
    with open("dummy_poc_repo.py", "w") as f:
        f.write(dummy_code)
        
    print("Building Graph...")
    pipeline.build_repo_graph("dummy_poc_repo.py")
    
    print("Running Smart Agent for 'process_users'...")
    try:
        summary = pipeline.summarize_with_agent(function_name="process_users")
        print("\n--- Final Summary ---\n")
        print(summary)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists("dummy_poc_repo.py"):
            os.remove("dummy_poc_repo.py")

if __name__ == "__main__":
    test_poc()
