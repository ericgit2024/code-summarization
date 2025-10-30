import ast
from datasets import load_dataset

def load_and_clean_dataset(dataset_name="code_x_glue_ct_code_to_text", subset="python"):
    """
    Loads the dataset, cleans it, and returns the filtered dataset.
    """
    dataset = load_dataset(dataset_name, subset)

    def is_valid_example(example):
        try:
            ast.parse(example['code'])
            return len(example['docstring'].strip()) > 0
        except SyntaxError:
            return False

    dataset = dataset.filter(is_valid_example)
    return dataset

if __name__ == "__main__":
    dataset = load_and_clean_dataset()
    print("Dataset splits after cleaning and filtering:")
    print(dataset)
    print("\nDataset Statistics:")
    for split in dataset.keys():
        print(f"  {split}: {len(dataset[split])} examples")
    print("\nFirst 5 examples from the training set:")
    print(dataset['train'][0:5])
