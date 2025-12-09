import torch
from transformers import AutoTokenizer, EncoderDecoderModel

def load_codegraphbert_model(model_id="microsoft/graphcodebert-base"):
    """
    Loads the GraphCodeBERT model as an Encoder-Decoder for summarization.
    Since GraphCodeBERT is an encoder-only model, we use EncoderDecoderModel
    to initialize a Seq2Seq architecture.

    Args:
        model_id (str): The Hugging Face model ID.
                        Default is "microsoft/graphcodebert-base".
    """
    print(f"Loading GraphCodeBERT model: {model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Initialize Encoder-Decoder model from the pre-trained encoder
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_id, model_id)

        # Set decoder configuration
        model.config.decoder_start_token_id = tokenizer.cls_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.vocab_size = model.config.encoder.vocab_size

        # Validating config
        model.config.eos_token_id = tokenizer.sep_token_id
        model.config.max_length = 512
        model.config.min_length = 56
        model.config.no_repeat_ngram_size = 3
        model.config.early_stopping = True
        model.config.length_penalty = 2.0
        model.config.num_beams = 4

    except Exception as e:
        raise RuntimeError(f"Failed to load GraphCodeBERT model/tokenizer: {e}") from e

    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.to("cuda")

    return model, tokenizer
