from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import torch
from PIL import Image

from .class_names import GTSRB_CLASS_NAMES_RU
from .config import BEST_WEIGHTS_PATH, CLASS_MAP_PATH
from .model import TrafficSignCNN
from .transforms import build_eval_transform, make_model_input_preview


@dataclass
class PredictionResult:
    label_id: int
    label_name: str
    confidence: float
    status: str
    top3: list[tuple[int, str, float]]
    roi_box: tuple[int, int, int, int] | None
    input_preview: Image.Image


class TrafficSignPredictor:
    def __init__(self, checkpoint_path: Path | None = None, class_map_path: Path | None = None) -> None:
        self.checkpoint_path = Path(checkpoint_path or BEST_WEIGHTS_PATH)
        self.class_map_path = Path(class_map_path or CLASS_MAP_PATH)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Best weights not found: {self.checkpoint_path}. Run training first (--mode train)."
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        self.input_size = int(checkpoint.get("input_size", 64))
        self.uncertain_threshold = float(checkpoint.get("uncertain_threshold", 0.60))

        class_map = checkpoint.get("class_map")
        if class_map is None:
            class_map = self._load_class_map_fallback()
        self.class_map = {int(k): str(v) for k, v in class_map.items()}

        num_classes = int(checkpoint.get("num_classes", len(self.class_map)))
        self.model = TrafficSignCNN(num_classes=num_classes)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.eval_transform = build_eval_transform(self.input_size)

    def _load_class_map_fallback(self) -> dict[int, str]:
        if self.class_map_path.exists():
            with self.class_map_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            return {int(k): str(v) for k, v in raw.items()}
        return dict(GTSRB_CLASS_NAMES_RU)

    @staticmethod
    def _clamp_roi(roi: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int] | None:
        x1, y1, x2, y2 = roi
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))

        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    @staticmethod
    def _status_from_confidence(confidence: float) -> str:
        if confidence >= 0.80:
            return "уверенно"
        if confidence >= 0.60:
            return "средне"
        return "сомнительно"

    @torch.no_grad()
    def predict(self, image: Image.Image, roi: tuple[int, int, int, int] | None = None) -> PredictionResult:
        image = image.convert("RGB")

        used_roi = None
        if roi is not None:
            clamped = self._clamp_roi(roi, image.width, image.height)
            if clamped is not None:
                used_roi = clamped
                image = image.crop((clamped[0], clamped[1], clamped[2], clamped[3]))

        preview = make_model_input_preview(image, self.input_size)
        tensor = self.eval_transform(image).unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        top_idx = np.argsort(-probs)[:3]
        top3: list[tuple[int, str, float]] = []
        for idx in top_idx:
            idx_i = int(idx)
            top3.append((idx_i, self.class_map.get(idx_i, f"Class {idx_i}"), float(probs[idx_i])))

        label_id = top3[0][0]
        confidence = top3[0][2]
        status = self._status_from_confidence(confidence)

        return PredictionResult(
            label_id=label_id,
            label_name=self.class_map.get(label_id, f"Class {label_id}"),
            confidence=confidence,
            status=status,
            top3=top3,
            roi_box=used_roi,
            input_preview=preview,
        )
