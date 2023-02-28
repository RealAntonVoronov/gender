import os
from argparse import ArgumentParser

import evaluate
import nltk
import numpy as np
import wandb
from nltk.tokenize import sent_tokenize
from transformers import (BioGptTokenizer, BioGptForCausalLM,
                          DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer)

from data import create_dataset

nltk.download('punkt')
rouge_score = evaluate.load("rouge")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model_name_or_path",
                        choices=['microsoft/biogpt', 'microsoft/biogpt-large'],
                        help="type of model to train.",
                        default='microsoft/biogpt')
    parser.add_argument("--dataset", required=True, choices=['biocarta', 'kegg', 'wp', 'pid', 'reactome'],
                        help="dataset for fine-tuning and evaluation")
    parser.add_argument("--data_dir", required=True, help='path to the folder with clusters and gene annotations')
    parser.add_argument("--n_permutations", default=10, type=int,
                        help="variable for data augmentation. "
                             "Defines how many genes permutations of the same cluster will be used for one description")
    parser.add_argument("--negative_frac", default=0.5, type=float,
                        help="Data augmentation and regularization variable. "
                             "Defines how many negative examples will be generated for a dataset. "
                             "E.g., negative_frac = 0.5 means that in the resulting dataset "
                             "there will be 1/3 of negative examples.")
    parser.add_argument("--negative_description_strategy", default='default', choices=['default'],
                        help="Strategy for generating descriptions of negative clusters. "
                             "Default strategy: annotating every cluster with "
                             "'This group of genes does not group into a meaningful cluster.'")
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

    train_dataset, eval_dataset = create_dataset(tokenizer=tokenizer,
                                                 dataset=args.dataset,
                                                 data_dir=args.data_dir,
                                                 n_permutations=args.n_permutations,
                                                 negative_frac=args.negative_frac,
                                                 negative_description_strategy=args.negative_description_strategy,
                                                 max_output_length=args.max_output_length,
                                                 seed=args.seed,
                                                 )

    collator = DataCollatorForSeq2Seq(tokenizer)
    effective_batch_size = args.train_batch_size * args.gradient_accumulation_steps
    n_epoch_steps = len(train_dataset) // effective_batch_size
    trainer_args = Seq2SeqTrainingArguments(
        output_dir=args.exp_name,
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
    np.random.seed(args.seed)
    main(args)
