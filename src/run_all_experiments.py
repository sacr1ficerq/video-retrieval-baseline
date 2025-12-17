#!/usr/bin/env python3
import subprocess
from pathlib import Path


EMBED_DIRS = {
    "clip": "./embeddings/clip/",
    "vlm": "./embeddings/vlm/",
}

MODELS = ["xclip", "siglip", "vlm"]
DATASETS = ["msrvtt", "vatex", "youcook2"]


def run_experiment(embed_dir, dataset, model):
    model_to_dir = {"xclip": "clip", "siglip": "clip", "vlm": "vlm"}
    actual_dir = EMBED_DIRS[model_to_dir[model]]

    cmd = [
        "python", "main.py",
        "--embed_dir", actual_dir,
        "--dataset", dataset,
        "--model", model
    ]

    subprocess.run(cmd)


def main():
    experiments = [
        ("xclip", "msrvtt"),
        ("xclip", "vatex"),
        ("xclip", "youcook2"),
        ("siglip", "msrvtt"),
        ("siglip", "vatex"),
        ("siglip", "youcook2"),
        ("vlm", "msrvtt"),
        ("vlm", "vatex"),
        ("vlm", "youcook2"),
    ]

    for model, dataset in experiments:
        run_experiment(None, dataset, model)


if __name__ == "__main__":
    main()
