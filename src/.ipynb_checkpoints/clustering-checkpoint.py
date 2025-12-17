import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score, 
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from sklearn.preprocessing import LabelEncoder


def cluster_embeddings(embeddings, n_clusters, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return labels, kmeans


def compute_supervised_metrics(true_labels, pred_labels):
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    return {"ARI": ari, "NMI": nmi}


def compute_unsupervised_metrics(embeddings, pred_labels):
    silhouette = silhouette_score(embeddings, pred_labels)
    calinski = calinski_harabasz_score(embeddings, pred_labels)
    davies = davies_bouldin_score(embeddings, pred_labels)

    return {
        "Silhouette": silhouette,
        "Calinski-Harabasz": calinski,
        "Davies-Bouldin": davies
    }


def evaluate_clustering(embeddings, categories=None, n_clusters=12):
    le = LabelEncoder()
    true_labels = le.fit_transform(categories)

    if categories is not None and len(np.unique(true_labels)) > 1:
        n_clusters = len(np.unique(true_labels))
        pred_labels, kmeans = cluster_embeddings(embeddings, n_clusters)

        supervised_metrics = compute_supervised_metrics(true_labels, pred_labels)
        unsupervised_metrics = compute_unsupervised_metrics(embeddings, pred_labels)

        metrics = {**supervised_metrics, **unsupervised_metrics}
        return metrics, n_clusters

    else:
        assert n_clusters is not None, "n_clusters required when categories is None"

        pred_labels, kmeans = cluster_embeddings(embeddings, n_clusters)
        unsupervised_metrics = compute_unsupervised_metrics(embeddings, pred_labels)

        return unsupervised_metrics, n_clusters


def evaluate_clustering_range(embeddings, k_range):
    results = {}

    for k in k_range:
        pred_labels, _ = cluster_embeddings(embeddings, k)
        metrics = compute_unsupervised_metrics(embeddings, pred_labels)
        results[k] = metrics

    return results