from torch.utils.data import Dataset


class GeneDataset(Dataset):
    def __init__(self,
                 clusters,
                 tokenizer,
                 genes_info,
                 max_input_length=400,
                 max_model_length=1024,
                 method='hard_prompt',
                 prompt='. Summarize: ',
                 training=True,
                 ):
        super().__init__()
        self.training = training
        self.method = method
        self.prompt = prompt
        self.tokenizer = tokenizer

        self.prompt_len = len(tokenizer(prompt, add_special_tokens=False)['input_ids'])
        max_output_length = max_model_length - max_input_length - self.prompt_len - 1
        print(f"max output length: {max_output_length}")

        id_to_description = {row['ncbi_id']: row['description'] for _, row in genes_info.iterrows()}

        texts = [". ".join([id_to_description.get(gene_id, 'unknown gene') for gene_id in cluster['genes']])
                 for _, cluster in clusters.iterrows()]

        self.input = tokenizer(texts, add_special_tokens=False,
                               truncation=True, max_length=max_input_length)['input_ids']
        if 'full' in clusters:
            self.output = tokenizer(clusters['full'].tolist(), add_special_tokens=False,
                                    truncation=True, max_length=max_output_length)['input_ids']
        else:
            self.output = None

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        if self.training:
            if self.method == 'hard_prompt':
                input_ids = self.input[idx] + self.tokenizer(self.prompt, add_special_tokens=False)['input_ids'] \
                            + self.output[idx] + [self.tokenizer.eos_token_id]
                labels = [-100 for _ in range(len(self.input[idx]) + self.prompt_len)]
                labels += self.output[idx] + [self.tokenizer.eos_token_id]
            else:
                # TODO! implement other methods (tunable prompt, ???)
                raise NotImplementedError
        else:
            input_ids = self.input[idx] + self.tokenizer(self.prompt, add_special_tokens=False)['input_ids']
            if self.output is not None:
                labels = self.output[idx] + [self.tokenizer.eos_token_id]
            else:
                return {"input_ids": input_ids,
                        "attention_mask": [1] * len(input_ids),
                        }

        return {"input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
                }