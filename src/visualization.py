import numpy as np
from sklearn.manifold import TSNE
import umap


def visualize_tsne(embeddings, perplexity=30, random_state=42):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    embeddings_2d = tsne.fit_transform(embeddings)
    return embeddings_2d


def visualize_umap(embeddings, n_neighbors=15, min_dist=0.1, random_state=42):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    embeddings_2d = reducer.fit_transform(embeddings)
    return embeddings_2d


def compute_rank(embeddings):
    return np.linalg.matrix_rank(embeddings)
