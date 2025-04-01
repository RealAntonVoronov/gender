import argparse
import json
import os

import pandas as pd
from openai import OpenAI
from tqdm import tqdm, trange
DEFAULT_GPT_PROMPT = "You are a helpful assistant."


def get_gpt_response(client, gpt_input):
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "system", "content": DEFAULT_GPT_PROMPT},
                  {"role": "user", "content": gpt_input},
                  ]
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["annotate", "structure_output"])
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--model", type=str, default="gpt-4o")
    args = parser.parse_args()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    if args.task == "annotate":
        input_data = pd.read_csv(args.input_file)
        genes_info = pd.read_csv("data/genes/genes_info.csv")

        id_to_symbol = {row['ncbi_id']: row['symbol'] for _, row in genes_info.iterrows()}
        inputs = []
        for _, row in input_data.iterrows():
            genes = eval(row['genes'])
            genes = [id_to_symbol[int(gene)] for gene in genes]
            inputs.append(genes)

        with open("prompts/annotate.txt") as f:
            template = f.read()

    elif args.task == "structure_output":
        with open(args.input_file) as f:
            inputs = f.readlines()

        with open("prompts/structure_output.txt") as f:
            template = f.read()

    else:
        raise NotImplementedError(f"Task {args.task} not implemented")

    if os.path.exists(args.output_file):
        with open(args.output_file) as f:
            gpt_responses = json.load(f)
    else:
        gpt_responses = []

    for i in trange(len(gpt_responses), len(inputs)):
        gpt_input = template.format(input=inputs[i])
        response = get_gpt_response(client, gpt_input)
        gpt_responses.append(response)

        with open(args.output_file, "w") as f:
            json.dump(gpt_responses, f)

        if i == 0:
            break
