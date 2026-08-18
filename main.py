from sentence_transformers import SentenceTransformer
from sentence_transformers.util import paraphrase_mining
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from umap import UMAP
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

    paraphrased = {}
    for entity, series in to_transform.items():
        paraphrased[entity] = paraphrase_miner(series)

    for entity, df in clustered.items():
        df.to_excel(f'outputs/{entity}_clustered.xlsx', index=False)

if __name__ == "__main__":
    main()
