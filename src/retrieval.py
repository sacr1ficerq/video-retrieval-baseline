import numpy as np
import faiss


def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-8)


def build_faiss_index(video_embeddings):
    video_embeddings = normalize_embeddings(video_embeddings)
    d = video_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(video_embeddings.astype(np.float32))
    return index


def search_videos(index, text_embeddings, k=10):
    text_embeddings = normalize_embeddings(text_embeddings)
    distances, indices = index.search(text_embeddings.astype(np.float32), k)
    return distances, indices


def compute_recall_at_k(indices, k_values=[1, 5, 10]):
    n = indices.shape[0]
    correct_indices = np.arange(n)

    recalls = {}
    for k in k_values:
        top_k = indices[:, :k]
        hits = np.any(top_k == correct_indices[:, None], axis=1)
        recalls[f"R@{k}"] = hits.mean()

    return recalls


def compute_median_rank(indices):
    n = indices.shape[0]
    correct_indices = np.arange(n)

    ranks = []
    for i in range(n):
        rank = np.where(indices[i] == correct_indices[i])[0]
        if len(rank) > 0:
            ranks.append(rank[0] + 1)
        else:
            ranks.append(indices.shape[1] + 1)

    return np.median(ranks)


def evaluate_retrieval(video_embeddings, text_embeddings, k_values=[1, 5, 10], 
                       advanced=False):
    index = build_faiss_index(video_embeddings)
    distances, indices = search_videos(index, text_embeddings, k=max(k_values))

    recalls = compute_recall_at_k(indices, k_values)
    median_rank = compute_median_rank(indices)

    metrics = {**recalls, "MedR": median_rank}

    if advanced:
        from advanced_metrics import compute_advanced_metrics
        adv_metrics = compute_advanced_metrics(distances, indices, k_values)
        metrics.update(adv_metrics)

    return metrics
