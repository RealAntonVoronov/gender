import pandas as pd
import numpy as np

MIN_CLUSTER_LEN_FOR_SPLIT = 10


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
                ):

    if isinstance(dataset, list):
        clusters = []
        for one_dataset in dataset:
            clusters.append(parse_dataset(one_dataset, data_dir=data_dir))
        clusters = pd.concat(clusters)
    else:
        clusters = parse_dataset(dataset, data_dir=data_dir)

    return clusters

def get_negative_clusters(clusters, negative_frac=0, placeholder_probability=1,
                          negative_description_strategy='default',
                          id_to_full_description=None):
    negative_clusters = []
    n_negative = int(negative_frac * len(clusters))
    if n_negative > 0:
        clusters_genes = []
        for _, row in clusters.iterrows():
            clusters_genes.extend(row['genes'])
        clusters_genes = set(clusters_genes)
        for i in range(n_negative):
            random_genes = np.random.permutation(list(clusters_genes))[:np.random.randint(5, 31)]
            if np.random.random() < placeholder_probability:
                description = sample_random_description(strategy=negative_description_strategy) 
            else:
                genes_subsample = np.random.choice(random_genes, np.random.randint(1, 4), replace=False)
                gene_descriptions = []
                for gene in genes_subsample:
                    col = np.random.choice(['summary_ncbi', 'summary_swissprot'])
                    gene_description_row = id_to_full_description.loc[id_to_full_description['ncbi_id'] == gene]
                    if len(gene_description_row) == 0:
                        gene_description = 'unknown gene'
                    else:
                        assert len(gene_description_row) == 1
                        gene_description = gene_description_row[col].values[0]
                    gene_descriptions.append(gene_description)
                description = ". ".join(gene_descriptions)
            negative_clusters.append({'name': f'negative cluster {i}', 'genes': random_genes,
                                      'brief': 'random negative cluster',
                                      'full': description})
            
    return negative_clusters


def get_short_clusters(clusters, n_short_subsamples=0):
    short_clusters = []
    for _, cluster in clusters.iterrows():
        cluster_len = len(cluster['genes'])
        if cluster_len > MIN_CLUSTER_LEN_FOR_SPLIT:
            subsample_size = round(0.5 * cluster_len)
            for seed in range(n_short_subsamples):
                np.random.seed(seed)
                new_genes = np.random.choice(cluster['genes'], subsample_size,
                                             replace=False,
                                             )
                short_clusters.append(pd_row_new_cluster(cluster['name'], new_genes,
                                                       cluster['brief'],
                                                       cluster['full']),
                                                       )

    return short_clusters


def augment_data(clusters,
                 id_to_full_description=None,
                 n_permutations=10,
                 n_short_subsamples=0,
                 negative_frac=0.3,
                 negative_description_strategy='default',
                 placeholder_probability=0.5,
                 ):
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
    short_clusters = get_short_clusters(clusters, n_short_subsamples)
    clusters = pd.concat([clusters, pd.DataFrame(short_clusters).reset_index(drop=True)])

    # add negative samples
    negative_clusters = get_negative_clusters(clusters, id_to_full_description,
                                             negative_frac, negative_description_strategy,
                                             placeholder_probability)
    clusters = pd.concat([clusters.reset_index(drop=True), pd.DataFrame(negative_clusters).reset_index(drop=True)])

    return clusters
