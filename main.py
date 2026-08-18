from sentence_transformers import SentenceTransformer
from sentence_transformers.util import paraphrase_mining
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans, AgglomerativeClustering
from umap import UMAP
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from openpyxl.drawing.image import Image as XLImage
import numpy as np
import pandas as pd

def embed_elements(elements:pd.Series) -> tuple[pd.Series, np.ndarray]:
    text = elements.reset_index(drop=True)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    vectors = model.encode(text.to_list(), show_progress_bar=True)
    return text, vectors


def cluster_kmeans(vectors:np.ndarray) -> np.ndarray:
    number_of_clusters = round(len(vectors) / 3)
    kmeans = KMeans(number_of_clusters, random_state=42)
    return kmeans.fit_predict(vectors)


def cluster_hdbscan(vectors:np.ndarray) -> np.ndarray:
    # Reduce number of components in vector to improve clustering in HDBSCAN
    reduced = UMAP(n_components=10, metric='cosine', random_state=42).fit_transform(vectors)
    hdbscan = HDBSCAN(metric='euclidean')
    return hdbscan.fit_predict(reduced)


def cluster_agglomerative(vectors: np.ndarray) -> np.ndarray:
    number_of_clusters = round(len(vectors) / 3)
    agglomerative = AgglomerativeClustering(n_clusters=number_of_clusters, metric="cosine", linkage='average')
    return agglomerative.fit_predict(vectors)


def build_linkage(vectors: np.ndarray) -> np.ndarray:
    # Matches the metric/linkage used by cluster_agglomerative.
    return linkage(vectors, method='average', metric='cosine')


def plot_dendrogram(linkage_matrix: np.ndarray, path: str, truncate_p: int = 50) -> None:
    # Truncated to the last `truncate_p` merges since full dendrograms are unreadable at this scale.
    plt.figure(figsize=(12, max(8, truncate_p * 0.25)))
    dendrogram(linkage_matrix, truncate_mode='lastp', p=truncate_p, show_contracted=True, orientation='right')
    plt.xlabel('Cosine distance')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def hierarchy_dataframe(text: pd.Series, linkage_matrix: np.ndarray) -> pd.DataFrame:
    # Each row is one merge step from scipy's linkage matrix: parent_id is the id
    # assigned to the new cluster formed by merging child_1 and child_2. Child ids
    # below n_leaves refer to original elements (resolved to their text below);
    # ids >= n_leaves refer to an earlier merge's parent_id.
    n_leaves = len(text)
    df = pd.DataFrame(linkage_matrix, columns=['child_1', 'child_2', 'distance', 'sample_count'])
    df['child_1'] = df['child_1'].astype(int)
    df['child_2'] = df['child_2'].astype(int)
    df['sample_count'] = df['sample_count'].astype(int)
    df.insert(0, 'parent_id', df.index + n_leaves)
    df['child_1_text'] = df['child_1'].map(lambda idx: text.iloc[idx] if idx < n_leaves else None)
    df['child_2_text'] = df['child_2'].map(lambda idx: text.iloc[idx] if idx < n_leaves else None)
    return df


def agglomerative_to_excel(agglomerated, hierarchies, linkages, export_dendrogram=True):
    with pd.ExcelWriter('outputs/agglomerative_clustered.xlsx', engine='openpyxl') as writer:
        for entity, df in agglomerated.items():
            df.to_excel(writer, sheet_name=entity, index=False)

        for entity, df in hierarchies.items():
            df.to_excel(writer, sheet_name=f'{entity}_hierarchy', index=False)

        if export_dendrogram:
            for entity, linkage_matrix in linkages.items():
                image_path = f'outputs/{entity}_dendrogram.png'
                plot_dendrogram(linkage_matrix, image_path)


def paraphrase_miner(elements:pd.Series) -> pd.DataFrame:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentences = elements.to_list()
    paraphrases = paraphrase_mining(model, sentences)
    for paraphrase in paraphrases[0:10]:
        score, i, j = paraphrase
        print("{} \t\t {} \t\t Score: {:.4f}".format(sentences[i], sentences[j], score))

    df = pd.DataFrame(paraphrases, columns=['score', 'i', 'j'])
    df['sentence_1'] = df['i'].map(lambda idx: sentences[idx])
    df['sentence_2'] = df['j'].map(lambda idx: sentences[idx])
    return df[['sentence_1', 'sentence_2', 'score']]


def main():
    df = pd.read_excel("all_merged_after_manual_llm_cleaned.xlsx")
    to_transform = {
        "threats": df['Cause'].dropna().drop_duplicates(),
        "consequences": df['Consequence'].dropna().drop_duplicates(),
        "control": df['Measure'].dropna().drop_duplicates(),
        }

    embedded = {}
    for entity, series in to_transform.items():
        text, vectors = embed_elements(series)
        embedded[entity] = (text, vectors)

    # clustered = {}
    # for entity, (text, vectors) in embedded.items():
        # clustered[entity] = pd.DataFrame({
            # 'text': text,
            # 'kmeans_cluster_id': cluster_kmeans(vectors),
            # 'hdbscan_cluster_id': cluster_hdbscan(vectors),
        # })

    # paraphrased = {}
    # for entity, series in to_transform.items():
        # paraphrased[entity] = paraphrase_miner(series)

    agglomerated = {}
    linkages = {}
    for entity, (text, vectors) in embedded.items():
        agglomerated[entity] = pd.DataFrame({
            'text': text,
            'cluster_id': cluster_agglomerative(vectors),
        }).sort_values('cluster_id')
        linkages[entity] = build_linkage(vectors)

    hierarchies = {
        entity: hierarchy_dataframe(text, linkages[entity])
        for entity, (text, _) in embedded.items()
    }

    agglomerative_to_excel(agglomerated, hierarchies, linkages)

    # for entity, df in clustered.items():
        # df.to_excel(f'outputs/{entity}_clustered.xlsx', index=False)

    # for entity, df in paraphrased.items():
        # df.to_excel(f'outputs/{entity}_paraphrased.xlsx', index=False)

if __name__ == "__main__":
    main()
