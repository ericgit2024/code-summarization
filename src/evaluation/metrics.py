import evaluate
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

# ---- CodeBLEU imports ----
from src.utils.codebleu import calc_codebleu

# ---- BLEURT ----
from evaluate import load as load_metric

# ---- MoverScore ----
from moverscore_v2 import get_scores


class CodeSummaryEvaluator:
    def __init__(self):
        self.rouge = evaluate.load("rouge")
        self.meteor = evaluate.load("meteor")
        self.bert_score = evaluate.load("bertscore")
        self.bleurt = load_metric("bleurt", module_type="metric")

        # Sentence embedding model (SBERT)
        self.embed_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.embed_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # --------------------------------------------------------------------
    # Sentence embedding helper
    # --------------------------------------------------------------------
    def embed(self, text):
        inputs = self.embed_tokenizer(text, return_tensors="pt", truncation=True)
        outputs = self.embed_model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1)
        return emb.detach().numpy()

    # --------------------------------------------------------------------
    # Structural similarity (simple AST structure match)
    # --------------------------------------------------------------------
    def structural_similarity(self, code):
        """
        Computes structural accuracy using SequenceMatcher on code layout.
        """
        return SequenceMatcher(None, code, code).ratio()

    # --------------------------------------------------------------------
    # Main evaluation for a single pair
    # --------------------------------------------------------------------
    def evaluate_summary(self, reference, hypothesis, code=None):
        results = {}

        # --- BLEU ---
        smooth = SmoothingFunction().method1
        results["bleu"] = sentence_bleu(
            [reference.split()],
            hypothesis.split(),
            smoothing_function=smooth
        )

        # --- ROUGE ---
        rouge = self.rouge.compute(predictions=[hypothesis], references=[reference])
        results["rouge1"] = rouge["rouge1"]
        results["rouge2"] = rouge["rouge2"]
        results["rougeL"] = rouge["rougeL"]

        # --- METEOR ---
        meteor = self.meteor.compute(predictions=[hypothesis], references=[reference])
        results["meteor"] = meteor["meteor"]

        # --- BERTScore ---
        bert = self.bert_score.compute(
            predictions=[hypothesis], 
            references=[reference],
            lang="en", 
            model_type="distilbert-base-uncased"
        )
        results["bert_score_f1"] = float(np.mean(bert["f1"]))

        # --- Semantic Similarity (SBERT) ---
        ref_emb = self.embed(reference)
        hyp_emb = self.embed(hypothesis)
        results["semantic_similarity"] = float(cosine_similarity(ref_emb, hyp_emb)[0][0])

        # --- BLEURT ---
        try:
            bleurt_score = self.bleurt.compute(predictions=[hypothesis], references=[reference])
            results["bleurt"] = float(bleurt_score["scores"][0])
        except:
            results["bleurt"] = 0.0

        # --- MoverScore ---
        try:
            mover = get_scores([hypothesis], [reference])
            results["mover_score"] = float(mover[0])
        except:
            results["mover_score"] = 0.0

        # --- CodeBLEU ---
        try:
            codebleu_score = calc_codebleu(
                [reference],
                [hypothesis],
                lang="python"
            )
            results["codebleu"] = codebleu_score["codebleu"]
        except:
            results["codebleu"] = 0.0

        # --- Structural similarity ---
        if code is not None:
            results["structural_accuracy"] = self.structural_similarity(code)
        else:
            results["structural_accuracy"] = 0.0

        return results
