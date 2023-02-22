import os
from argparse import ArgumentParser

import evaluate
import nltk
nltk.download('punkt')
import numpy as np
import pandas as pd
import wandb
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (BioGptTokenizer, BioGptForCausalLM,
                          DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer)


rouge_score = evaluate.load("rouge")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model_name_or_path",
                        choices=['microsoft/biogpt', 'microsoft/biogpt-large'],
                        help="type of model to train.",
                        default='microsoft/biogpt')
    parser.add_argument("--dataset", required=True, choices=['biocarta', 'kegg', 'wp', 'pid', 'reactome'],
                        help="dataset for fine-tuning and evaluation")
    parser.add_argument("--data_dir", )
    parser.add_argument("--max_output_length", help="max length of the generated descriptions", default=128, type=int)
    parser.add_argument("--train_batch_size", help="train batch size per 1 GPU", default=4, type=int)
    parser.add_argument("--eval_batch_size", help="validation batch size per 1 GPU", default=4, type=int)
    parser.add_argument("--learning_rate", "-lr", help="final learning rate for AdamW optimizer with warmup",
                        default=1e-5, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="use this to increase effective train batch size")
    parser.add_argument("--num_train_epochs", default=8, type=int, help="number of training epochs")
    parser.add_argument("--num_beams", default=3, type=int, help="number of beams to use in beam search decoding")
    parser.add_argument("--seed", help='seed for reproducibility', default=37, type=int)
    parser.add_argument("--exp_name", default=None,
                        help="name of the experiment for WandB logging. Also used as a name for the output dir")
    parser.add_argument("--fp16", action='store_true',
                        help='use this option to train in float16, for better performance')
    parser.add_argument("--deepspeed", action='store_true',
                        help='use this option to train using Deepspeed library, for better performance')
    args = parser.parse_args()

    return args


class GeneData(Dataset):
    def __init__(self, clusters, tokenizer, genes_info,
                 max_input_length=768,
                 max_output_length=128,
                 max_model_length=1024,
                 ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_model_length = max_model_length
        assert max_input_length + max_output_length + 3 < max_model_length

        texts = [". ".join([genes_info[genes_info['ncbi_id'] == gene_id]['description'].item()
                            if gene_id in genes_info['ncbi_id'].value_counts() else "unknown gene"
                            for gene_id in cluster['genes']])
                 for _, cluster in clusters.iterrows()]

        self.input = tokenizer(texts, truncation=True, max_length=max_input_length)['input_ids']
        self.output = tokenizer(list(clusters['full'].values),
                                truncation=True, max_length=max_output_length)['input_ids']

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        input_ids = self.input[idx] + self.tokenizer("summarize: ")['input_ids'] + self.output[idx]
        labels = input_ids
        return {"input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
                }


def create_dataset(tokenizer, dataset, max_output_length=128, seed=37):
    clusters = {'name': [], 'genes': []}

    with open(f"../data/bio/clusters/{dataset}.txt") as inp:
        for line in inp.readlines():
            name, _, *genes = line.split("\t")
            clusters['name'].append(name)
            clusters['genes'].append([int(x.strip()) for x in genes])

    clusters = pd.DataFrame(clusters)
    cluster_desc = pd.read_csv(f"../data/bio/clusters/{dataset}_descriptions.csv")
    cluster_desc.columns = ['name', 'brief', 'full']
    clusters = pd.merge(clusters, cluster_desc)
    # TODO! drop rows where `full` is empty

    genes_info = pd.read_csv("../data/bio/genes/genes_info.csv")
    # TODO! add augmentation

    # TODO! add negative samples

    train, val = train_test_split(clusters, random_state=seed, test_size=0.2)
    train = GeneData(clusters=train, tokenizer=tokenizer, genes_info=genes_info,
                     max_output_length=max_output_length)
    val = GeneData(clusters=val, tokenizer=tokenizer, genes_info=genes_info,
                   max_output_length=max_output_length)

    return train, val


def compute_metrics(eval_pred, tokenizer, rouge_score):
    predictions, labels = eval_pred
    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    # Compute ROUGE scores
    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract the median scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}


def main(args):
    model = BioGptForCausalLM.from_pretrained(args.model_name_or_path)
    tokenizer = BioGptTokenizer.from_pretrained(args.model_name_or_path)

    train_dataset, eval_dataset = create_dataset(tokenizer=tokenizer, dataset=args.dataset,
                                                 max_output_length=args.max_output_length,
                                                 seed=args.seed)

    collator = DataCollatorForSeq2Seq(tokenizer)
    effective_batch_size = args.train_batch_size * args.gradient_accumulation_steps

    trainer_args = Seq2SeqTrainingArguments(
        output_dir=args.exp_name,
        overwrite_output_dir=True,
        eval_steps=10,
        save_steps=100,
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=args.num_train_epochs,
        predict_with_generate=False,
        logging_steps=1,
        seed=args.seed,
        fp16=args.fp16,
        deepspeed=args.deepspeed,
        report_to='wandb',
    )

    trainer = Seq2SeqTrainer(
        model,
        trainer_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    #TODO! evaluate


if __name__ == '__main__':
    args = parse_args()
    wandb.init(entity='antonvoronov', project='GENDER')
    if args.exp_name is not None:
        wandb.run.name = args.exp_name
    else:
        args.exp_name = 'out'
    os.makedirs(args.exp_name, exist_ok=True)
    main(args)
