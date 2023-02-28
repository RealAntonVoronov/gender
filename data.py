import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class GeneData(Dataset):
    def __init__(self, clusters, tokenizer, genes_info,
                 max_input_length=768,
                 max_output_length=128,
                 max_model_length=1024,
                 method='hard_prompt',
                 prompt='summarize: ',
                 ):
        super().__init__()
        self.method = method
        self.prompt = prompt
        self.tokenizer = tokenizer
        self.max_model_length = max_model_length
        assert max_input_length + max_output_length + 3 < max_model_length
        id_to_description = {row['ncbi_id']: row['description'] for _, row in genes_info.iterrows()}
        texts = [". ".join([id_to_description.get(gene_id, 'unknown gene') for gene_id in cluster['genes']])
                 for _, cluster in clusters.iterrows()]

        self.input = tokenizer(texts, truncation=True, max_length=max_input_length)['input_ids']
        self.output = tokenizer(list(clusters['full'].values),
                                truncation=True, max_length=max_output_length)['input_ids']

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        if self.method == 'hard_prompt':
            input_ids = self.input[idx] + self.tokenizer(self.prompt)['input_ids'] + self.output[idx]
        else:
            # TODO! implement other methods (tunable prompt, ???)
            raise NotImplementedError
        labels = input_ids
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


def create_dataset(tokenizer, dataset, data_dir,
                   n_permutations=10,
                   negative_frac=0.3,
                   negative_description_strategy='default',
                   max_output_length=128,
                   seed=37,
                   ):
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

    # add augmentation
    new_clusters = []
    for i, row in clusters.iterrows():
        new_clusters.extend([{'name': row['name'], 'genes': np.random.permutation(row.genes),
                              'brief': row.brief, 'full': row.full}
                             for _ in range(n_permutations)])
    clusters = pd.concat([clusters, pd.DataFrame(new_clusters).reset_index(drop=True)])


    # add negative samples
    negative_clusters = []
    n_negative = int(negative_frac * len(clusters))

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

    genes_info = pd.read_csv(f"{data_dir}/genes/genes_info.csv")

    train, val = train_test_split(clusters, random_state=seed, test_size=0.2)
    train = GeneData(clusters=train, tokenizer=tokenizer, genes_info=genes_info,
                     max_output_length=max_output_length)
    val = GeneData(clusters=val, tokenizer=tokenizer, genes_info=genes_info,
                   max_output_length=max_output_length)

    return train, val
