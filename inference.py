from argparse import ArgumentParser
from tqdm import tqdm

import torch
import pandas as pd
from transformers import BioGptForCausalLM, BioGptTokenizer, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader

from data import GeneDataset


def generate(model, tokenizer, dataloader, max_length=1024, num_beams=1):
    # TODO! return more than 1 hypotheses
    results = []
    for batch in tqdm(dataloader):
        inputs = batch['input_ids']
        output = model.generate(input_ids=inputs.to(model.device),
                                attention_mask=batch['attention_mask'].to(model.device),
                                max_length=max_length,
                                num_beams=num_beams,
                                do_sample=False,
                                )
        # replace prefix part in output with special tokens to skip them
        output[:, :len(inputs[0])] = tokenizer.pad_token_id
        results.extend(tokenizer.batch_decode(output, skip_special_tokens=True))
    return results


def inference(model, tokenizer, test_dataset, eval_batch_size=1, max_model_length=1024, num_beams=1):
    test_collator = DataCollatorForSeq2Seq(tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=eval_batch_size, collate_fn=test_collator, shuffle=False)

    test_preds = generate(model, tokenizer, test_dataloader, max_length=max_model_length, num_beams=num_beams)
    return test_preds


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_path",
                        help="Checkpoint to a trained model. If not specified, "
                             "microsoft/biogpt checkpoint from HuggingFace Hub is used.",
                        default='microsoft/biogpt',
    )
    parser.add_argument("--test_file", help="Path to a test file for prediction.", default='data/test.tsv')
    parser.add_argument("-o", "--output_file", help="Path to store model predictions", default='output.txt')
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search decoding")
    args = parser.parse_args()
    return args


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BioGptForCausalLM.from_pretrained(args.checkpoint_path)
    tokenizer = BioGptTokenizer.from_pretrained(args.checkpoint_path, padding_side='left')
    model.to(device)
    max_model_length = model.config.max_position_embeddings

    test_dataframe = pd.read_csv(args.test_file, sep='\t')

    genes_info = pd.read_csv("data/genes/genes_info.csv")
    symbol_to_id = {row['symbol']: row['ncbi_id'] for _, row in genes_info.iterrows()}

    test_clusters = []
    for cluster_idx, group in test_dataframe.groupby('cluster'):
        test_clusters.append({'name': f'test cluster {cluster_idx}',
                              'genes': [symbol_to_id.get(symbol, -1) for symbol in group['gene'].tolist()],
                              })
    test_clusters = pd.DataFrame(test_clusters)

    test_dataset = GeneDataset(clusters=test_clusters,
                               tokenizer=tokenizer,
                               genes_info=genes_info,
                               max_input_length=400,
                               training=False,
                               )
    test_preds = inference(model, tokenizer, test_dataset, eval_batch_size=args.eval_batch_size,
                           max_model_length=max_model_length, num_beams=args.num_beams,
                           )
    with open(args.output_file, 'w') as f_out:
        f_out.writelines('\n'.join(test_preds))

if __name__ == '__main__':
    args = parse_args()
    main(args)