import pandas as pd
import json
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


model_dir = "results/biocarta_kegg-large-perm20-neg0.01-add_rouge/"
if __name__ == "__main__":
    with open(f"{model_dir}/test_preds", "r") as f:
        last_preds = f.readlines()

    labels = pd.read_csv("data/test/val_clusters.csv")["full"]
    print(compute_rouge(last_preds, labels))
