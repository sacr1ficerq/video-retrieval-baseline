#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from data_loader import load_dataset_embeddings
from visualization import visualize_tsne, visualize_umap


def plot_embeddings_2d(embeddings_2d, categories, title, output_path=None):
    plt.figure(figsize=(12, 10))

    unique_categories = np.unique(categories)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_categories)))

    for idx, category in enumerate(unique_categories):
        mask = categories == category
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   c=[colors[idx]], label=category, alpha=0.6, s=30)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()
