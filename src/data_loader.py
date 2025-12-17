import pickle
from pathlib import Path
import numpy as np


def load_embeddings(path):
    with open(path, "rb") as f:
        df = pickle.load(f)
    return df


def extract_features(df, video_key="video_emb", text_key="text_emb"):
    video_embs = np.vstack([np.array(emb).mean(axis=0) if len(emb.shape) > 1 else emb 
                            for emb in df[video_key]])
    text_embs = np.vstack([np.array(emb).mean(axis=0) if len(emb.shape) > 1 else emb 
                           for emb in df[text_key]])
    return video_embs, text_embs


def load_dataset_embeddings(base_dir, dataset_name, model_name):
    path = Path(base_dir) / f"{dataset_name}_{model_name}_embeddings.pkl"
    assert path.exists(), f"File not found: {path}"
    df = load_embeddings(path)
    # print(f"Columns: {df.columns}")
    if "video_emb" in df.columns and "text_emb" in df.columns:
        video_embs, text_embs = extract_features(df)
    elif "generated_embedding" in df.columns and "embedding" in df.columns:
        video_embs, text_embs = extract_features(df, "embedding", "generated_embedding")
    elif "embedding" in df.columns:
        if "mae" in model_name:
            video_embs = np.vstack(df["embedding"])
            text_embs = None
        else:
            text_embs = np.vstack(df["embedding"])
            video_embs = None
    else:
        raise ValueError(f"Unknown embedding format in {path}")

    categories = df["category"].values if "category" in df.columns else None
    return video_embs, text_embs, categories, df
