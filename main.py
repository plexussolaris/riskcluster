from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from umap import UMAP
import numpy as np
import pandas as pd
from typing import List

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


def main():
    df = pd.read_excel("all_merged_after_manual_llm.xlsx")
    to_transform = {
        "threats": df['Cause'].dropna().drop_duplicates(),
        "consequences": df['Consequence'].dropna().drop_duplicates(),
        "control": df['Measure'].dropna().drop_duplicates(),
        }

    embedded = {}
    for entity, series in to_transform.items():
        text, vectors = embed_elements(series)
        embedded[entity] = (text, vectors)

    clustered = {}
    for entity, (text, vectors) in embedded.items():
        clustered[entity] = pd.DataFrame({
            'text': text,
            'kmeans_cluster_id': cluster_kmeans(vectors),
            'hdbscan_cluster_id': cluster_hdbscan(vectors),
        })

    for entity, df in clustered.items():
        df.to_excel(f'outputs/{entity}_clustered.xlsx')

if __name__ == "__main__":
    main()
