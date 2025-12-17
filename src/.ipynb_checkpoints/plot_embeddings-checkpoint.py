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


def main():
    parser = argparse.ArgumentParser(description="Plot embeddings with t-SNE/UMAP")
    parser.add_argument("--embed_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--method", type=str, default="both", choices=["tsne", "umap", "both"])
    parser.add_argument("--output_dir", type=str, default="./plots")
    parser.add_argument("--perplexity", type=int, default=30)
    parser.add_argument("--n_neighbors", type=int, default=15)

    args = parser.parse_args()

    video_embs, text_embs, categories, df = load_dataset_embeddings(
        args.embed_dir, args.dataset, args.model
    )

    assert categories is not None, "Categories required for plotting"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if args.method in ["tsne", "both"]:
        print("Running t-SNE...")
        embeddings_2d = visualize_tsne(video_embs, perplexity=args.perplexity)
        title = f"{args.model.upper()} on {args.dataset.upper()} (t-SNE)"
        output_path = output_dir / f"{args.dataset}_{args.model}_tsne.png"
        plot_embeddings_2d(embeddings_2d, categories, title, output_path)

    if args.method in ["umap", "both"]:
        print("Running UMAP...")
        embeddings_2d = visualize_umap(video_embs, n_neighbors=args.n_neighbors)
        title = f"{args.model.upper()} on {args.dataset.upper()} (UMAP)"
        output_path = output_dir / f"{args.dataset}_{args.model}_umap.png"
        plot_embeddings_2d(embeddings_2d, categories, title, output_path)


if __name__ == "__main__":
    main()
