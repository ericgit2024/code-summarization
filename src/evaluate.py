import torch
from torch.utils.data import DataLoader
import evaluate
from .data_loader import load_and_clean_dataset
from .prompt_generator import get_combined_structural_prompt
from .retrieval import get_encoder_model, encode_codes, build_faiss_index, retrieve_similar_codes
from .model_handler import load_lora_model
from .train import construct_prompt

def generate_predictions(prompts, model, tokenizer, max_length=128):
    """
    Generates summaries for a batch of prompts using the fine-tuned model.
    """
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def compute_metrics(predictions, references, bleu_metric, rouge_metric, meteor_metric):
    """
    Computes BLEU, ROUGE, and METEOR scores.
    """
    formatted_references = [[ref] for ref in references]

    bleu_score = bleu_metric.compute(predictions=predictions, references=formatted_references)
    rouge_score_result = rouge_metric.compute(predictions=predictions, references=references)
    meteor_score_result = meteor_metric.compute(predictions=predictions, references=references)

    return {
        "bleu": bleu_score["bleu"],
        "rouge": rouge_score_result,
        "meteor": meteor_score_result["meteor"]
    }

def prepare_evaluation_data(example, retrieval_index, dataset, encoder_model):
    """
    Prepares a single example for evaluation.
    """
    structural_prompt = get_combined_structural_prompt(example['code'])
    retrieved_codes, retrieved_docstrings, _ = retrieve_similar_codes(example['code'], retrieval_index, dataset, encoder_model, k=3)
    final_prompt = construct_prompt(structural_prompt, example['code'], retrieved_codes, retrieved_docstrings)
    return {'prompt': final_prompt, 'reference': example['docstring']}

if __name__ == '__main__':
    dataset = load_and_clean_dataset()
    encoder_model = get_encoder_model()

    train_codes = dataset['train']['code']
    train_embeddings = encode_codes(train_codes, encoder_model)
    index = build_faiss_index(train_embeddings)

    model = load_lora_model("deepseek-ai/deepseek-coder-1.3b-base")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base")

    test_dataset_subset = dataset['test'].select(range(100))
    processed_test_dataset = test_dataset_subset.map(lambda example: prepare_evaluation_data(example, index, dataset, encoder_model))

    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")

    def custom_collate_fn(batch):
        return {key: [item[key] for item in batch] for key in batch[0]}

    eval_dataloader = DataLoader(processed_test_dataset, batch_size=16, collate_fn=custom_collate_fn)

    all_predictions = []
    all_references = []

    model.eval()
    with torch.no_grad():
        for batch in eval_dataloader:
            prompts = batch['prompt']
            references = batch['reference']

            predictions = generate_predictions(prompts, model, tokenizer)

            all_predictions.extend(predictions)
            all_references.extend(references)

    final_metrics = compute_metrics(all_predictions, all_references, bleu_metric, rouge_metric, meteor_metric)

    print("\nEvaluation Results:")
    print(final_metrics)
