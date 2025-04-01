import nltk
import evaluate
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')
rouge_score = evaluate.load("rouge")

def compute_rouge(preds, labels):
    preds = [sent_tokenize(pred) for pred in preds]
    labels = [sent_tokenize(label) for label in labels]
    decoded_preds = ["\n".join(pred) for pred in preds]
    decoded_labels = ["\n".join(label) for label in labels]
    
    # Compute ROUGE scores
    result = rouge_score.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )
    # Extract the median scores
    result = {key: value * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}
