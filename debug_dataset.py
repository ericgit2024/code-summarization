
from src.data.dataset import load_and_process_dataset

try:
    train_ds = load_and_process_dataset(split="train")
    print(f"Train dataset size: {len(train_ds)}")
    print(f"Train example 0: {train_ds[0]['code'][:50]}...")
except Exception as e:
    print(f"Train load failed: {e}")

try:
    val_ds = load_and_process_dataset(split="validation")
    print(f"Validation dataset size: {len(val_ds)}")
    print(f"Validation example 0: {val_ds[0]['code'][:50]}...")
except Exception as e:
    print(f"Validation load failed: {e}")
