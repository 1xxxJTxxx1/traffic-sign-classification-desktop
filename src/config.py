from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
ASSETS_DIR = PROJECT_ROOT / "assets"
REFERENCE_ICONS_DIR = ASSETS_DIR / "reference_icons"

MANIFEST_PATH = PROCESSED_DATA_DIR / "manifest.csv"
DATASET_META_PATH = PROCESSED_DATA_DIR / "dataset_meta.json"
CLASS_MAP_PATH = PROCESSED_DATA_DIR / "class_map.json"
DATASET_PREVIEW_PATH = REPORTS_DIR / "dataset_preview.png"

BEST_WEIGHTS_PATH = MODELS_DIR / "traffic_sign_cnn_best.pth"
LAST_WEIGHTS_PATH = MODELS_DIR / "traffic_sign_cnn_last.pth"
TRAIN_HISTORY_PATH = REPORTS_DIR / "train_history.csv"
TRAIN_REPORT_PATH = REPORTS_DIR / "training_report.json"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"


@dataclass(frozen=True)
class DatasetSettings:
    val_ratio: float = 0.15
    demo_samples_per_class: int = 25
    seed: int = 42


@dataclass(frozen=True)
class TrainingSettings:
    input_size: int = 64
    batch_size: int = 96
    epochs: int = 35
    lr: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    num_workers: int = 0
    early_stopping_patience: int = 7
    require_gpu: bool = False


NORMALIZATION_MEAN = [0.3403, 0.3121, 0.3214]
NORMALIZATION_STD = [0.2724, 0.2608, 0.2669]


def ensure_project_dirs() -> None:
    for path in (
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        ARTIFACTS_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        ASSETS_DIR,
        REFERENCE_ICONS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def dataset_settings_to_dict(settings: DatasetSettings) -> dict:
    return asdict(settings)


def training_settings_to_dict(settings: TrainingSettings) -> dict:
    return asdict(settings)


