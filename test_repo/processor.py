
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
