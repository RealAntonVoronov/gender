import os
from functools import partial
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.model_selection import train_test_split
from transformers import (BioGptTokenizer, BioGptForCausalLM,
                          Trainer, TrainingArguments)

from data_utils import create_data, augment_data
from dataset import GeneDataset
from evaluation import compute_rouge
from inference import inference


def collate_fn(tokenizer, batch):
    pad_ids = {'input_ids': tokenizer.pad_token_id, 'attention_mask': 0, 'labels': -100}

    max_len = np.max([len(x['input_ids']) for x in batch])
    start_positions = [max_len - len(x['input_ids']) for x in batch]

    return {k: torch.tensor([[pad_ids[k] for _ in range(start_positions[i])] + batch[i][k]
                             for i in range(len(batch))], dtype=torch.long)
            for k in pad_ids}


def parse_args():
    parser = ArgumentParser()
    # general arguments
    parser.add_argument("-m", "--model_name_or_path",
                        choices=['microsoft/biogpt', 'microsoft/biogpt-large'],
                        help="Type of model to train.",
                        default='microsoft/biogpt')
    parser.add_argument("--dataset", required=True, choices=['biocarta', 'kegg', 'wp', 'pid', 'reactome'], nargs='+',
                        help="Dataset for fine-tuning and evaluation")
    parser.add_argument("--data_dir", required=True, help='Path to the folder with clusters and gene annotations.')
    parser.add_argument("--method", default='hard_prompt',
                        help='Method of fine-tuning. '
                             'Hard-prompt requires specifying prompt which is later used in every example. '
                             'Soft-prompt adds learnable tokens to the model vocabulary and adapts them to the task '
                             'during fine-tuning.')
    parser.add_argument("--prompt", default="summarize: ",
                        help="Prompt that is used to specify the task for the model "
                             "if the fine-tuning method is `hard-prompt`.")
    # generation arguments
    parser.add_argument("--num_beams", default=3, type=int, help="Number of beams to use in beam search decoding.")
    # dataset arguments
    parser.add_argument("--n_permutations", default=20, type=int,
                        help="Data augmentation variable. "
                             "Defines how many gene permutations of the same cluster will be used for one description.")
    parser.add_argument("--negative_frac", default=0.05, type=float,
                        help="Data augmentation and regularization variable. "
                             "Defines how many negative examples will be generated for a dataset. "
                             "E.g., negative_frac = 0.5 means that in the resulting dataset "
                             "there will be 1/3 of negative examples.")
    parser.add_argument("--negative_description_strategy", default='default', choices=['default'],
                        help="Strategy for generating descriptions of negative clusters. "
                             "Default strategy: annotating every cluster with "
                             "'This group of genes does not group into a meaningful cluster.'")
    parser.add_argument("--n_short_subsamples", default=0, type=int)
    parser.add_argument("--max_input_length", help="Max length of the input sequence.", default=400, type=int)
    # training arguments
    parser.add_argument("--train_batch_size", help="Train batch size per 1 GPU.", default=4, type=int)
    parser.add_argument("--eval_batch_size", help="Validation batch size per 1 GPU.", default=4, type=int)
    parser.add_argument("--learning_rate", "-lr", help="Final learning rate for AdamW optimizer with warmup.",
                        default=1e-5, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Use this to increase effective train batch size.")
    parser.add_argument("--num_train_epochs", default=4, type=int, help="Number of training epochs.")
    parser.add_argument("--seed", help='Seed for reproducibility.', default=37, type=int)
    # utility arguments
    parser.add_argument("--exp_name", default=None,
                        help="Name of the experiment for WandB logging. Also used as a name for the output dir.")
    parser.add_argument("--fp16", action='store_true',
                        help='Use this option to train in float16, for better performance.')
    parser.add_argument("--deepspeed", action='store_true',
                        help='Use this option to train using Deepspeed library, for better performance.')
    args = parser.parse_args()

    return args


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds
    # Replace -100 in the preds as we can't decode them
    preds = np.where(labels != -100, preds, tokenizer.eos_token_id)
    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.eos_token_id)
    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    rouge = compute_rouge(decoded_preds, decoded_labels)
    wandb.log({"preds_0": decoded_preds[0], "labels_0": decoded_labels[0]})
    return rouge


def train(args):
    model = BioGptForCausalLM.from_pretrained(args.model_name_or_path)
    tokenizer = BioGptTokenizer.from_pretrained(args.model_name_or_path, padding_side='left')
    model.cuda()
    max_model_length = model.config.max_position_embeddings

    clusters = create_data(dataset=args.dataset, data_dir=args.data_dir)
    train_clusters, val_clusters = train_test_split(clusters, random_state=args.seed, test_size=0.2)
    val_clusters.to_csv(f"{args.data_dir}/test/val_clusters.csv", index=False)
    genes_full_description = pd.read_csv(f"{args.data_dir}/genes/gene_descriptions.csv")
    train_clusters = augment_data(train_clusters,
                                  id_to_full_description=genes_full_description,
                                  n_permutations=args.n_permutations,
                                  negative_frac=args.negative_frac,
                                  negative_description_strategy=args.negative_description_strategy,
                                  n_short_subsamples=args.n_short_subsamples,
                                  placeholder_probability=1,
                                  )
    print(len(train_clusters), len(val_clusters))
    genes_info = pd.read_csv(f"{args.data_dir}/genes/genes_info.csv")
    
    train_dataset = GeneDataset(clusters=train_clusters,
                                tokenizer=tokenizer,
                                genes_info=genes_info,
                                max_input_length=args.max_input_length,
                                max_model_length=max_model_length,
                                method=args.method,
                                prompt=args.prompt,
                                )
    eval_dataset = GeneDataset(clusters=val_clusters,
                               tokenizer=tokenizer,
                               genes_info=genes_info,
                               max_input_length=args.max_input_length,
                               max_model_length=max_model_length,
                               method=args.method,
                               prompt=args.prompt,
                               )

    effective_batch_size = args.train_batch_size * args.gradient_accumulation_steps
    n_epoch_steps = len(train_dataset) // effective_batch_size

    trainer_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        eval_steps=n_epoch_steps//3,
        save_steps=n_epoch_steps//3,
        evaluation_strategy='steps',
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=args.num_train_epochs,
        logging_steps=1,
        seed=args.seed,
        fp16=args.fp16,
        deepspeed=args.deepspeed,
        report_to=None,
    )

    trainer = Trainer(
        model,
        trainer_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=partial(collate_fn, tokenizer),
        tokenizer=tokenizer,
        compute_metrics=partial(compute_metrics, tokenizer=tokenizer),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    trainer.train()
    # evaluate in the end
    test_dataset = GeneDataset(clusters=val_clusters,
                               tokenizer=tokenizer,
                               genes_info=genes_info,
                               max_input_length=args.max_input_length,
                               method=args.method,
                               prompt=args.prompt,
                               training=False,
                               )
    test_preds = inference(model, tokenizer, test_dataset, args.eval_batch_size, max_model_length, args.num_beams)
    test_labels = tokenizer.batch_decode(test_dataset.output, skip_special_tokens=True)
    assert len(test_preds) == len(test_labels)
    with open(f"{args.output_dir}/test_preds", "w") as f_preds, open(f"{args.output_dir}/test_labels", "w") as f_labels:
        for i in range(len(test_preds)):
            f_preds.writelines(test_preds[i]+'\n')
            f_labels.writelines(test_labels[i] + '\n')

    metrics = compute_rouge(test_preds, test_labels)
    print(f"ROUGE scores: {metrics}")
    for k, v in metrics.items():
        wandb.run.summary[k] = v


if __name__ == '__main__':
    args = parse_args()
    wandb.init(entity='antonvoronov', project='GENDER')
    if args.exp_name is not None:
        wandb.run.name = args.exp_name
        pass
    else:
        args.exp_name = 'out'
    args.output_dir = os.path.join('results', args.exp_name)
    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    train(args)
