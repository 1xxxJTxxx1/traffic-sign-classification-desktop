from __future__ import annotations

from pathlib import Path
import hashlib
import json
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import (
    BEST_WEIGHTS_PATH,
    CLASS_MAP_PATH,
    CONFUSION_MATRIX_PATH,
    LAST_WEIGHTS_PATH,
    MANIFEST_PATH,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    TRAIN_HISTORY_PATH,
    TRAIN_REPORT_PATH,
    TrainingSettings,
    dump_json,
    ensure_project_dirs,
    now_iso,
    training_settings_to_dict,
)
from .datasets import ManifestDataset
from .model import TrafficSignCNN, count_trainable_params, get_model_signature
from .transforms import build_eval_transform, build_train_transform


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_device(require_gpu: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_gpu:
        raise RuntimeError("CUDA is not available, and --require-gpu is enabled.")
    print("[Train] CUDA not available. Falling back to CPU training.")
    return torch.device("cpu")


def _sha256_of_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def build_dataset_fingerprint() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("Dataset manifest not found. Run --mode prepare first.")

    df = pd.read_csv(MANIFEST_PATH, usecols=["split", "label"])
    split_counts = {str(k): int(v) for k, v in df.groupby("split").size().to_dict().items()}

    class_map_hash = None
    if CLASS_MAP_PATH.exists():
        class_map_hash = _sha256_of_file(CLASS_MAP_PATH)

    return {
        "manifest_sha256": _sha256_of_file(MANIFEST_PATH),
        "manifest_size": int(MANIFEST_PATH.stat().st_size),
        "class_map_sha256": class_map_hash,
        "row_count": int(len(df)),
        "num_classes": int(df["label"].nunique()),
        "split_counts": split_counts,
    }


def inspect_weights_compatibility(
    settings: TrainingSettings,
    weights_path: Path | None = None,
) -> dict:
    weights_path = Path(weights_path or BEST_WEIGHTS_PATH)

    reasons: list[str] = []
    details: dict = {}

    if not MANIFEST_PATH.exists():
        reasons.append("dataset_manifest_missing")
        return {
            "weights_path": str(weights_path),
            "needs_retrain": True,
            "reasons": reasons,
            "details": details,
        }

    dataset_fingerprint = build_dataset_fingerprint()
    current_model_signature = get_model_signature()

    details["current_dataset_fingerprint"] = dataset_fingerprint
    details["current_model_signature"] = current_model_signature

    if not weights_path.exists():
        reasons.append("weights_missing")
        return {
            "weights_path": str(weights_path),
            "needs_retrain": True,
            "reasons": reasons,
            "details": details,
        }

    try:
        checkpoint = torch.load(weights_path, map_location="cpu")
    except Exception as exc:
        reasons.append("weights_unreadable")
        details["weights_read_error"] = str(exc)
        return {
            "weights_path": str(weights_path),
            "needs_retrain": True,
            "reasons": reasons,
            "details": details,
        }

    checkpoint_dataset_fingerprint = checkpoint.get("dataset_fingerprint")
    checkpoint_model_signature = checkpoint.get("model_signature")
    checkpoint_num_classes = checkpoint.get("num_classes")

    details["checkpoint_meta"] = {
        "trained_at": checkpoint.get("trained_at"),
        "model_name": checkpoint.get("model_name"),
        "model_signature": checkpoint_model_signature,
        "num_classes": checkpoint_num_classes,
        "dataset_fingerprint": checkpoint_dataset_fingerprint,
    }

    if checkpoint_dataset_fingerprint is None:
        reasons.append("checkpoint_without_dataset_fingerprint")
    else:
        if checkpoint_dataset_fingerprint.get("manifest_sha256") != dataset_fingerprint.get("manifest_sha256"):
            reasons.append("dataset_manifest_changed")
        if checkpoint_dataset_fingerprint.get("class_map_sha256") != dataset_fingerprint.get("class_map_sha256"):
            reasons.append("dataset_class_map_changed")

    if checkpoint_model_signature is None:
        reasons.append("checkpoint_without_model_signature")
    elif checkpoint_model_signature != current_model_signature:
        reasons.append("architecture_changed")

    if int(checkpoint_num_classes or -1) != int(dataset_fingerprint["num_classes"]):
        reasons.append("num_classes_changed")

    return {
        "weights_path": str(weights_path),
        "needs_retrain": len(reasons) > 0,
        "reasons": reasons,
        "details": details,
    }


def _load_class_map() -> dict[int, str]:
    if CLASS_MAP_PATH.exists():
        with CLASS_MAP_PATH.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return {int(k): str(v) for k, v in raw.items()}
    return {i: f"Class {i}" for i in range(43)}


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
) -> tuple[float, float]:
    model.train()
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels, _ in tqdm(loader, desc="[Train]", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        losses.append(float(loss.item()))
        preds = torch.argmax(logits, dim=1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    return float(np.mean(losses)), float(accuracy_score(y_true, y_pred))


@torch.no_grad()
def _eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels, _ in tqdm(loader, desc="[Val]", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)

        losses.append(float(loss.item()))
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    return (
        float(np.mean(losses)),
        float(accuracy_score(y_true, y_pred)),
        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    )


@torch.no_grad()
def _predict_split(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[int], list[int]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, labels, _ in tqdm(loader, desc="[Test]", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(labels.tolist())
    return y_true, y_pred


def _plot_confusion(cm: np.ndarray, class_map: dict[int, str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    labels = [str(i) for i in sorted(class_map.keys())]
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted class id",
        ylabel="True class id",
        title="Confusion Matrix (Test)",
    )
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def train_model(
    settings: TrainingSettings,
    force_retrain: bool = False,
    seed: int = 42,
) -> dict:
    ensure_project_dirs()

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("Dataset manifest not found. Run --mode prepare first.")

    if BEST_WEIGHTS_PATH.exists() and not force_retrain:
        compatibility = inspect_weights_compatibility(settings, BEST_WEIGHTS_PATH)
        if not compatibility["needs_retrain"]:
            print(f"[Train] Best weights already exist: {BEST_WEIGHTS_PATH}")
            print("[Train] Skip training. Use --force-retrain to train again.")
            return {
                "skipped": True,
                "reason": "best_weights_are_fresh",
                "best_weights": str(BEST_WEIGHTS_PATH),
                "compatibility": compatibility,
            }

        print("[Train] Existing weights are outdated. Retraining is required.")
        for reason in compatibility["reasons"]:
            print(f"[Train] Retrain reason: {reason}")

    set_seed(seed)
    device = _resolve_device(settings.require_gpu)
    print(f"[Train] Device: {device}")

    dataset_fingerprint = build_dataset_fingerprint()
    model_signature = get_model_signature()

    train_ds = ManifestDataset(MANIFEST_PATH, split="train", transform=build_train_transform(settings.input_size))
    val_ds = ManifestDataset(MANIFEST_PATH, split="val", transform=build_eval_transform(settings.input_size))
    test_ds = ManifestDataset(MANIFEST_PATH, split="test", transform=build_eval_transform(settings.input_size))

    class_map = _load_class_map()
    num_classes = len(class_map)

    train_loader = DataLoader(
        train_ds,
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=settings.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = TrafficSignCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=settings.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=settings.epochs)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    history: list[dict] = []

    print(f"[Train] Train samples: {len(train_ds)}")
    print(f"[Train] Val samples:   {len(val_ds)}")
    print(f"[Train] Test samples:  {len(test_ds)}")
    print(f"[Train] Num classes:   {num_classes}")
    print(f"[Train] Parameters:    {count_trainable_params(model):,}")

    for epoch in range(1, settings.epochs + 1):
        train_loss, train_acc = _train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, val_f1 = _eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        lr = float(optimizer.param_groups[0]["lr"])

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1_macro": val_f1,
            "lr": lr,
        }
        history.append(row)

        print(
            f"[Train] Epoch {epoch:02d}/{settings.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}% val_f1={val_f1:.4f} | lr={lr:.6f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "num_classes": num_classes,
            "class_map": class_map,
            "input_size": settings.input_size,
            "normalization": {"mean": NORMALIZATION_MEAN, "std": NORMALIZATION_STD},
            "settings": training_settings_to_dict(settings),
            "trained_at": now_iso(),
            "best_val_f1": best_val_f1,
            "uncertain_threshold": 0.60,
            "model_name": "TrafficSignCNN-v1",
            "model_signature": model_signature,
            "dataset_fingerprint": dataset_fingerprint,
        }
        LAST_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, LAST_WEIGHTS_PATH)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            checkpoint["best_val_f1"] = best_val_f1
            torch.save(checkpoint, BEST_WEIGHTS_PATH)
            print(f"[Train] New best weights saved: {BEST_WEIGHTS_PATH}")
        else:
            patience_counter += 1

        if patience_counter >= settings.early_stopping_patience:
            print(f"[Train] Early stopping at epoch {epoch} (patience={settings.early_stopping_patience})")
            break

    hist_df = pd.DataFrame(history)
    TRAIN_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    hist_df.to_csv(TRAIN_HISTORY_PATH, index=False)

    best_ckpt = torch.load(BEST_WEIGHTS_PATH, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    y_true, y_pred = _predict_split(model, test_loader, device)
    test_acc = float(accuracy_score(y_true, y_pred))
    test_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    labels_sorted = sorted(class_map.keys())
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    _plot_confusion(cm, class_map, CONFUSION_MATRIX_PATH)

    class_names = [class_map[i] for i in labels_sorted]
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels_sorted,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_payload = {
        "created_at": now_iso(),
        "model_name": "TrafficSignCNN-v1",
        "model_signature": model_signature,
        "dataset_fingerprint": dataset_fingerprint,
        "best_epoch": best_epoch,
        "best_val_f1_macro": best_val_f1,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
        "num_classes": num_classes,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "weights_path": str(BEST_WEIGHTS_PATH),
        "history_path": str(TRAIN_HISTORY_PATH),
        "confusion_matrix_path": str(CONFUSION_MATRIX_PATH),
        "settings": training_settings_to_dict(settings),
        "classification_report": report_dict,
    }
    dump_json(TRAIN_REPORT_PATH, report_payload)

    print("[Train] Training complete.")
    print(f"[Train] Best epoch: {best_epoch}")
    print(f"[Train] Test accuracy: {test_acc*100:.2f}%")
    print(f"[Train] Test F1 macro: {test_f1:.4f}")

    return report_payload
