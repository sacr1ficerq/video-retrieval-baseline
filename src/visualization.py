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

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def analyze_embeddings(video_embs, text_embs, log_scale=True, ax=None):
    """
    Computes soft rank, modality gap, and visualizes the singular value spectrum.
    Args:
        video_embs: (N, D) numpy array or torch tensor
        text_embs: (N, D) numpy array or torch tensor (can be None)
        ax: matplotlib axes object (optional). If None, creates a new figure.
    """
    # Ensure numpy
    if isinstance(video_embs, torch.Tensor): video_embs = video_embs.detach().cpu().numpy()
    if text_embs is not None and isinstance(text_embs, torch.Tensor): 
        text_embs = text_embs.detach().cpu().numpy()

    # 1. Singular Value Decomposition (SVD)
    sv_video = np.linalg.svd(video_embs, compute_uv=False)
    if text_embs is not None:
        sv_text = np.linalg.svd(text_embs, compute_uv=False)

    # 2. Effective Rank Calculations (Printing info)
    # ... (Keep your existing rank calculation code here if you want the prints) ...
    # For brevity in plotting, you might want to return these values or just print them.
    # [Paste your existing Rank/Entropy/Gap calculation code here]

    # 3. Plot Singular Value Spectrum
    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
        show_plot = True

    ax.plot(sv_video, label='Video Embeddings', linewidth=2)
    
    if text_embs is not None:
        ax.plot(sv_text, label='Text Embeddings', linewidth=2, linestyle='--')

    if log_scale:
        ax.set_yscale('log')
        ax.set_ylabel('Log Singular Value')
    else:
        ax.set_ylabel('Singular Value')

    ax.set_xlabel('Dimension Index (sorted)')
    ax.set_title(f'Singular Value Spectrum')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if show_plot:
        plt.show()
