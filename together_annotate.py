import os
import json

from together import Together
from tqdm import trange

from argparse import ArgumentParser
import pandas as pd

def get_model_response(client, model_name, model_input, max_tokens=400, temperature=0.0):
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": model_input}],
        max_tokens=max_tokens,
        temperature=temperature,
        n=1,
    )
    return response.choices[0].message.content

API_KEY = None
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    client = Together(api_key=API_KEY)

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

    if os.path.exists(args.output_file):
        with open(args.output_file) as f:
            model_responses = json.load(f)
    else:
        model_responses = []

    for i in trange(len(model_responses), len(inputs)):
        model_input = template.format(input=inputs[i])
        model_response = get_model_response(client, args.model_name, model_input)
        model_responses.append(model_response)

        with open(args.output_file, "w") as f:
            json.dump(model_responses, f)
