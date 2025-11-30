import os
import logging
from src.model.inference import InferencePipeline

# Setup logging to see Agent internal thoughts
logging.basicConfig(level=logging.INFO)

def create_dummy_repo():
    os.makedirs("test_repo", exist_ok=True)
    
    # File 1: Database Utils
    with open("test_repo/db_utils.py", "w") as f:
        f.write('''
def connect_db(connection_string):
    """
    Establishes a secure connection to the PostgreSQL database using the provided string.
    Handles retry logic and SSL handshake.
    """
    pass

def save_record(data):
    """
    Persists the data record to the 'audit_log' table.
    Encrypts sensitive fields before saving.
    """
    pass
''')

    # File 2: Main Logic
    with open("test_repo/processor.py", "w") as f:
        f.write('''
from db_utils import connect_db, save_record

def process_transaction(tx_data):
    """
    Processes a financial transaction.
    """
    conn = connect_db("postgres://...")
    
    # Validate
    if tx_data['amount'] > 10000:
        print("High value transaction")
        
    save_record(tx_data)
    return True
''')

def main():
    print("--- Setting up Test Repo ---")
    create_dummy_repo()
    
    print("\n--- Initializing Pipeline ---")
    # Initialize pipeline
    pipeline = InferencePipeline(repo_path="test_repo")
    
    # We want to see the agent in action.
    # We will summarize 'process_transaction'.
    # The initial context from RepoGraph SHOULD include db_utils functions if it works well.
    # But let's see if the Agent uses them or asks for more.
    
    print("\n--- Running Agentic Summarization ---")
    summary = pipeline.summarize_with_agent(function_name="process_transaction")
    
    print("\n\n=== FINAL SUMMARY ===")
    print(summary)
    
    # Clean up
    # import shutil
    # shutil.rmtree("test_repo")

if __name__ == "__main__":
    main()
