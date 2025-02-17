import pandas as pd
import numpy as np
from torch.utils.data import Dataset

MIN_CLUSTER_LEN_FOR_SPLIT = 10

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

        prompt_len = len(tokenizer(prompt, add_special_tokens=False)['input_ids'])
        max_output_length = max_model_length - max_input_length - prompt_len - 1
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
                labels = [-100 for _ in range(len(self.input[idx]))]
                labels += self.tokenizer(self.prompt, add_special_tokens=False)['input_ids'] + \
                          self.output[idx] + [self.tokenizer.eos_token_id]
            else:
                # TODO! implement other methods (tunable prompt, ???)
                raise NotImplementedError
        else:
            input_ids = self.input[idx] + self.tokenizer(self.prompt, add_special_tokens=False)['input_ids']
            if self.output is not None:
                labels = self.output[idx]
            else:
                return {"input_ids": input_ids,
                        "attention_mask": [1] * len(input_ids),
                        }

        return {"input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
                }


def sample_random_description(strategy):
    if strategy == 'default':
        return "This group of genes does not group into a meaningful cluster."
    else:
        # TODO! add other strategies (templates, paraphrasing)
        raise NotImplementedError

def parse_dataset(dataset, data_dir='data'):
    clusters = {'name': [], 'genes': []}
    
    with open(f"{data_dir}/clusters/{dataset}.txt") as inp:
        for line in inp.readlines():
            name, _, *genes = line.split("\t")
            clusters['name'].append(name)
            clusters['genes'].append([int(x.strip()) for x in genes])

    clusters = pd.DataFrame(clusters)
    cluster_desc = pd.read_csv(f"{data_dir}/clusters/{dataset}_descriptions.csv")
    cluster_desc.columns = ['name', 'brief', 'full']
    clusters = pd.merge(clusters, cluster_desc)
    clusters = clusters.drop(clusters[clusters['full'] == '\xa0'].index)
    clusters.reset_index(drop=True, inplace=True)

    return clusters

def pd_row_new_cluster(name, genes, brief_description, full_description):
    return {'name': name, 'genes': genes,
            'brief': brief_description, 'full': full_description,
            }

def create_data(dataset,
                data_dir,
                n_permutations=10,
                negative_frac=0.3,
                negative_description_strategy='default',
                n_short_subsamples=0,
                ):

    if isinstance(dataset, list):
        clusters = []
        for one_dataset in dataset:
            clusters.append(parse_dataset(one_dataset, data_dir=data_dir))
        clusters = pd.concat(clusters)
    else:
        clusters = parse_dataset(dataset, data_dir=data_dir)

    # add augmentation
    if n_permutations > 0:
        new_clusters = []
        for i, row in clusters.iterrows():
            for _ in range(n_permutations):
                new_clusters.append(pd_row_new_cluster(row.name,
                                                       np.random.permutation(row['genes']),
                                                       row['brief'], row['full'],
                                                       ))
        clusters = pd.concat([clusters, pd.DataFrame(new_clusters).reset_index(drop=True)])

    # augmentation for better short clusters
    new_clusters = []
    for _, cluster in clusters.iterrows():
        cluster_len = len(cluster['genes'])
        if cluster_len > MIN_CLUSTER_LEN_FOR_SPLIT:
            subsample_size = round(0.5 * cluster_len)
            for seed in range(n_short_subsamples):
                np.random.seed(seed)
                new_genes = np.random.choice(cluster['genes'], subsample_size,
                                             replace=False,
                                             )
                new_clusters.append(pd_row_new_cluster(cluster['name'], new_genes,
                                                       cluster['brief'],
                                                       cluster['full']),
                                                       )
    clusters = pd.concat([clusters, pd.DataFrame(new_clusters).reset_index(drop=True)])

    # add negative samples
    negative_clusters = []
    n_negative = int(negative_frac * len(clusters))
    if n_negative > 0:
        clusters_genes = []
        for _, row in clusters.iterrows():
            clusters_genes.extend(row['genes'])
        clusters_genes = set(clusters_genes)
        for i in range(n_negative):
            random_genes = np.random.permutation(list(clusters_genes))[:np.random.randint(5, 31)]
            negative_clusters.append({'name': f'negative cluster {i}', 'genes': random_genes,
                                      'brief': 'random negative cluster',
                                      'full': sample_random_description(strategy=negative_description_strategy)})

        clusters = pd.concat([clusters.reset_index(drop=True), pd.DataFrame(negative_clusters).reset_index(drop=True)])

    return clusters
