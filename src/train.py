from transformers import TrainingArguments, Trainer
from .data_loader import load_and_clean_dataset
from .prompt_generator import get_combined_structural_prompt
from .retrieval import get_encoder_model, encode_codes, build_faiss_index, retrieve_similar_codes
from .model_handler import load_model_and_tokenizer, get_lora_config, apply_lora
import torch

def construct_prompt(structural_prompt, query_code, retrieved_codes, retrieved_docstrings):
    """
    Combines structural prompt, retrieved examples, and target code into a single prompt string.
    """
    prompt = f"Structural Prompt:\n{structural_prompt}\n\n"
    for i in range(len(retrieved_codes)):
        prompt += f"Retrieved Code:\n{retrieved_codes[i]}\n"
        prompt += f"Retrieved Docstring:\n{retrieved_docstrings[i]}\n\n"
    prompt += f"Code to Summarize:\n{query_code}\n\nSummary:"
    return prompt

def prepare_finetuning_dataset(example, retrieval_index, dataset, encoder_model, tokenizer):
    """
    Prepares a single example for fine-tuning.
    """
    structural_prompt = get_combined_structural_prompt(example['code'])
    retrieved_codes, retrieved_docstrings, _ = retrieve_similar_codes(example['code'], retrieval_index, dataset, encoder_model, k=3)

    prompt = construct_prompt(structural_prompt, example['code'], retrieved_codes, retrieved_docstrings)
    summary = example['docstring']

    tokenized_prompt = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
    tokenized_summary = tokenizer(summary, truncation=True, padding="max_length", max_length=128)

    example['input_ids'] = tokenized_prompt['input_ids']
    example['attention_mask'] = tokenized_prompt['attention_mask']
    example['labels'] = tokenized_summary['input_ids']

    return example

if __name__ == '__main__':
    dataset = load_and_clean_dataset()
    encoder_model = get_encoder_model()

    train_codes = dataset['train']['code']
    train_embeddings = encode_codes(train_codes, encoder_model)
    index = build_faiss_index(train_embeddings)

    model, tokenizer = load_model_and_tokenizer("deepseek-ai/deepseek-coder-1.3b-base")
    lora_config = get_lora_config()
    model = apply_lora(model, lora_config)

    train_dataset_subset = dataset['train'].select(range(10))

    # We need to pass the index, dataset, and encoder_model to the map function
    # The simplest way to do this is with a lambda function
    processed_train_dataset = train_dataset_subset.map(
        lambda example: prepare_finetuning_dataset(example, index, dataset, encoder_model, tokenizer),
        remove_columns=['id', 'repo', 'path', 'func_name', 'original_string', 'language', 'code_tokens', 'docstring_tokens', 'sha', 'url', 'code', 'docstring']
    )

    processed_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    training_args = TrainingArguments(
        output_dir="./codet5_lora_finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=3e-4,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_train_dataset,
    )

    trainer.train()
