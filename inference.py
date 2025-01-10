from argparse import ArgumentParser
from tqdm import tqdm

import torch
import pandas as pd
from transformers import BioGptForCausalLM, BioGptTokenizer, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader

from data import GeneDataset, parse_dataset


def generate(model, tokenizer, batch, max_length=1024, num_beams=1):
    # TODO! return more than 1 hypotheses
    inputs = batch['input_ids']
    output = model.generate(input_ids=inputs.to(model.device),
                            attention_mask=batch['attention_mask'].to(model.device),
                            max_length=max_length,
                            num_beams=num_beams,
                            do_sample=False,
                            )
    # replace prefix part in output with special tokens to skip them
    output[:, :len(inputs[0])] = tokenizer.pad_token_id
    return tokenizer.batch_decode(output, skip_special_tokens=True)


def inference(model, tokenizer, test_dataset, eval_batch_size=1, max_model_length=1024, num_beams=1, res_path="out.txt"):
    test_collator = DataCollatorForSeq2Seq(tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=eval_batch_size, collate_fn=test_collator, shuffle=False)

    results = []
    for batch in tqdm(test_dataloader):
        test_preds = generate(model, tokenizer, batch, max_length=max_model_length, num_beams=num_beams)
        results.extend(test_preds)
        with open(res_path, "w") as f:
            f.writelines("\n".join(results))
    
    return results


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_path",
                        help="Checkpoint to a trained model. If not specified, "
                             "microsoft/biogpt checkpoint from HuggingFace Hub is used.",
                        default='microsoft/biogpt',
    )
    parser.add_argument("--test_dataset", help="name of dataset from data/clusters")
    parser.add_argument("--test_file", help="Path to a test file for prediction.", default='data/test.tsv')
    parser.add_argument("-o", "--output_file", help="Path to store model predictions", default='output.txt')
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search decoding")
    parser.add_argument("--max_input_length", type=int, default=512)
    args = parser.parse_args()
    return args


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BioGptForCausalLM.from_pretrained(args.checkpoint_path,
                                              torch_dtype=torch.bfloat16,
                                              )
    tokenizer = BioGptTokenizer.from_pretrained(args.checkpoint_path, padding_side='left')
    model.to(device)
    max_model_length = model.config.max_position_embeddings

    genes_info = pd.read_csv("data/genes/genes_info.csv")
    if args.test_dataset is not None:
        test_clusters = parse_dataset(args.test_dataset)
        test_clusters.to_csv("input.csv", index=False)
    else:
        if args.test_file.endswith('.tsv'):
            sep = '\t'
        else:
            sep = ','
        test_dataframe = pd.read_csv(args.test_file, sep=sep)
        test_dataframe = test_dataframe.loc[test_dataframe['cluster'] != 'Singleton']
        test_dataframe['cluster'] = test_dataframe['cluster'].astype(int)

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
                               max_input_length=args.max_input_length,
                               max_model_length=max_model_length,
                               training=False,
                               )
    test_preds = inference(model, tokenizer, test_dataset, eval_batch_size=args.eval_batch_size,
                           max_model_length=max_model_length, num_beams=args.num_beams,
                           res_path=args.output_file,
                           )

if __name__ == '__main__':
    args = parse_args()
    main(args)
