import numpy as np
import torch
import torch.nn.functional as F


def compute_softerrank(distances, temperature=0.1):
    neg_distances = -distances / temperature
    softmax_scores = F.softmax(torch.from_numpy(neg_distances).float(), dim=1)
    ranks = torch.arange(1, distances.shape[1] + 1).float()
    soft_ranks = (softmax_scores * ranks).sum(dim=1)
    return soft_ranks.numpy()


def compute_average_precision(indices, correct_idx):
    relevant_positions = np.where(indices == correct_idx)[0]
    if len(relevant_positions) == 0:
        return 0.0

    position = relevant_positions[0] + 1
    return 1.0 / position


def compute_mean_average_precision(indices):
    n = indices.shape[0]
    correct_indices = np.arange(n)

    aps = []
    for i in range(n):
        ap = compute_average_precision(indices[i], correct_indices[i])
        aps.append(ap)

    return np.mean(aps)


def compute_ndcg_at_k(indices, k=10):
    n = indices.shape[0]
    correct_indices = np.arange(n)

    ndcg_scores = []
    for i in range(n):
        relevance = (indices[i, :k] == correct_indices[i]).astype(float)

        if relevance.sum() == 0:
            ndcg_scores.append(0.0)
            continue

        dcg = relevance[0] + np.sum(relevance[1:] / np.log2(np.arange(2, k + 1)))
        idcg = 1.0
        ndcg_scores.append(dcg / idcg)

    return np.mean(ndcg_scores)


def compute_mrr(indices):
    n = indices.shape[0]
    correct_indices = np.arange(n)

    reciprocal_ranks = []
    for i in range(n):
        rank_positions = np.where(indices[i] == correct_indices[i])[0]
        if len(rank_positions) > 0:
            reciprocal_ranks.append(1.0 / (rank_positions[0] + 1))
        else:
            reciprocal_ranks.append(0.0)

    return np.mean(reciprocal_ranks)


def compute_advanced_metrics(distances, indices, k_values=[5, 10]):
    metrics = {}

    median_softerrank = np.median(compute_softerrank(distances))
    metrics["SoftMedR"] = median_softerrank

    metrics["MAP"] = compute_mean_average_precision(indices)
    metrics["MRR"] = compute_mrr(indices)

    for k in k_values:
        metrics[f"NDCG@{k}"] = compute_ndcg_at_k(indices, k)

    return metrics
