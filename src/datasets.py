from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ManifestDataset(Dataset):
    def __init__(self, manifest_path: Path, split: str, transform=None) -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        df = pd.read_csv(self.manifest_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No samples for split='{split}' in {self.manifest_path}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = Path(row["path"])
        label = int(row["label"])

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, label, str(img_path)


def read_manifest_counts(manifest_path: Path) -> Tuple[int, dict]:
    df = pd.read_csv(manifest_path)
    split_counts = df.groupby("split").size().to_dict()
    return len(df), {str(k): int(v) for k, v in split_counts.items()}


def collect_class_distribution(manifest_path: Path, split: str) -> List[Tuple[int, int]]:
    df = pd.read_csv(manifest_path)
    part = df[df["split"] == split]
    counts = part.groupby("label").size().to_dict()
    return sorted([(int(k), int(v)) for k, v in counts.items()], key=lambda x: x[0])
