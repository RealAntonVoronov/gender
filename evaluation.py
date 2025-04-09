import argparse
import json

import evaluate
import nltk
import pandas as pd
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds_file", required=True)
    parser.add_argument("--labels_file", required=False)
    args = parser.parse_args()
    with open(args.preds_file) as f:
        preds = json.load(f)

    labels = pd.read_csv("data/test/val_clusters.csv")["full"].tolist()

    for k, v in compute_rouge(preds, labels).items():
        print(f"{k}: {v.item()}")
