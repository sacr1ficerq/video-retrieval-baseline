import torch
from transformers import VideoMAEImageProcessor, VideoMAEModel
from torchvision.io import read_video
import numpy as np
from tqdm.notebook import tqdm
import pickle
import os

# Инициализация процессора и модели
processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base", device_map="auto")


# Функция для чтения видео и подготовки кадров
def load_video(path, num_frames=16):
    video, audio, info = read_video(path, pts_unit="sec")
    total_frames = video.shape[0]
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = video[indices].permute(0, 3, 1, 2)  # -> [T, C, H, W]
    return frames


# Функция для получения эмбеддинга
def get_video_embedding(video_path):
    frames = load_video(video_path)  # [T, C, H, W]
    inputs = processor(list(frames), return_tensors="pt").to("cuda:6")
    with torch.no_grad():
        outputs = model(**inputs)
        # outputs.last_hidden_state: [1, num_patches, hidden_size]
        embedding = outputs.last_hidden_state.mean(
            dim=1
        ).squeeze()  # усреднение по патчам
    return embedding


prefix = "/content"

datasets = {
    "msrvtt": "MSRVTT_videos",
    "youcook2": "YouCook2_videos",
    "vatex": "VATEX_videos",
}
for dataset, directory in datasets.items():
    with open(os.path.join(prefix, directory, f"{dataset}_dataset.pkl"), "rb") as f:
        df = pickle.load(f)

    embeddings = []
    for vp in tqdm(df["video_path"]):
        embeddings.append(
            get_video_embedding(os.path.join("/content", vp)).cpu().numpy()
        )
    df["embedding"] = embeddings

    with open(f"videomae_{dataset}_embeddings", "wb") as f:
        pickle.dump(df, f)

    print(df.head())
