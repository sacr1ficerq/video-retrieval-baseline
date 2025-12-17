#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np

from data_loader import load_dataset_embeddings
from clustering import evaluate_clustering
from retrieval import evaluate_retrieval
from visualization import compute_rank


MODELS = {
    "xclip": "xclip",
    "siglip": "siglip",
    "vlm": "",
}

DATASETS = ["msrvtt", "vatex", "youcook2"]


def run_evaluation(embed_dir, dataset, model, mode):
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset.upper()} | Model: {model.upper()} | Mode: {mode.upper()}")
    print('='*60)

    video_embs, text_embs, categories, df = load_dataset_embeddings(embed_dir, dataset, model)

    print(f"Video embeddings: {video_embs.shape}")
    if text_embs is not None:
        print(f"Text embeddings: {text_embs.shape}")

    if mode in ["all", "clustering"] and categories is not None:
        print("\n--- Clustering Metrics ---")
        metrics, n_clusters = evaluate_clustering(video_embs, categories)
        print(f"Number of clusters: {n_clusters}")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    if mode in ["all", "retrieval"] and text_embs is not None:
        print("\n--- Retrieval Metrics ---")
        metrics = evaluate_retrieval(video_embs, text_embs, k_values=[1, 5, 10])
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    if mode in ["all", "rank"]:
        print("\n--- Embedding Space Analysis ---")
        rank = compute_rank(video_embs)
        print(f"Matrix rank (video): {rank}/{video_embs.shape[1]}")
        if text_embs is not None:
            rank_text = compute_rank(text_embs)
            print(f"Matrix rank (text): {rank_text}/{text_embs.shape[1]}")


def main():
    parser = argparse.ArgumentParser(description="Video Retrieval Evaluation")
    parser.add_argument("--embed_dir", type=str, required=True, help="Directory with embeddings")
    parser.add_argument("--dataset", type=str, choices=DATASETS, required=True)
    parser.add_argument("--model", type=str, choices=list(MODELS.keys()), required=True)
    parser.add_argument("--mode", type=str, default="all", 
                        choices=["all", "clustering", "retrieval", "rank"])

    args = parser.parse_args()

    model_suffix = MODELS[args.model]
    run_evaluation(args.embed_dir, args.dataset, model_suffix, args.mode)


if __name__ == "__main__":
    main()
