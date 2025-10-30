from .data_loader import load_and_clean_dataset
from .prompt_generator import get_combined_structural_prompt
from .retrieval import get_encoder_model, encode_codes, build_faiss_index, retrieve_similar_codes
from .model_handler import load_lora_model
from .train import construct_prompt
from .evaluate import generate_predictions
from transformers import AutoTokenizer

if __name__ == '__main__':
    dataset = load_and_clean_dataset()
    encoder_model = get_encoder_model()

    train_codes = dataset['train']['code']
    train_embeddings = encode_codes(train_codes, encoder_model)
    index = build_faiss_index(train_embeddings)

    model = load_lora_model("deepseek-ai/deepseek-coder-1.3b-base")
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base")

    sample_index = 0
    sample_test_example = dataset['test'][sample_index]

    sample_structural_prompt = get_combined_structural_prompt(sample_test_example['code'])
    sample_retrieved_codes, sample_retrieved_docstrings, _ = retrieve_similar_codes(sample_test_example['code'], index, dataset, encoder_model, k=3)

    sample_final_prompt = construct_prompt(
        sample_structural_prompt,
        sample_test_example['code'],
        sample_retrieved_codes,
        sample_retrieved_docstrings
    )

    sample_generated_summary = generate_predictions([sample_final_prompt], model, tokenizer)

    print("--- Sample Code ---")
    print(sample_test_example['code'])
    print("\n--- Generated Summary ---")
    print(sample_generated_summary[0])
    print("\n--- Reference Summary ---")
    print(sample_test_example['docstring'])
