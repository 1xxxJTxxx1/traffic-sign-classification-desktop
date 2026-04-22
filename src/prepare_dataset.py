from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import csv
import random
import shutil

import pandas as pd
from PIL import Image, ImageDraw
from torchvision.datasets import GTSRB
from tqdm import tqdm

from .class_names import GTSRB_CLASS_NAMES_RU
from .config import (
    CLASS_MAP_PATH,
    DATASET_META_PATH,
    DATASET_PREVIEW_PATH,
    MANIFEST_PATH,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REFERENCE_ICONS_DIR,
    DatasetSettings,
    dump_json,
    ensure_project_dirs,
    now_iso,
)

IMAGE_EXTENSIONS = {".ppm", ".png", ".jpg", ".jpeg", ".bmp"}


def _locate_raw_paths(raw_dir: Path) -> tuple[Path, Path, Path]:
    root = raw_dir / "gtsrb"
    if not root.exists():
        raise FileNotFoundError(f"Raw folder not found: {root}")

    train_dir_candidates = [p for p in root.rglob("Training") if p.is_dir()]
    test_dir_candidates = [p for p in root.rglob("Images") if p.is_dir() and "Final_Test" in str(p)]
    test_csv_candidates = [p for p in root.rglob("GT-final_test.csv") if p.is_file()]

    if not train_dir_candidates:
        raise FileNotFoundError("Could not locate GTSRB training directory")
    if not test_dir_candidates:
        raise FileNotFoundError("Could not locate GTSRB test images directory")
    if not test_csv_candidates:
        raise FileNotFoundError("Could not locate GT-final_test.csv")

    return train_dir_candidates[0], test_dir_candidates[0], test_csv_candidates[0]


def _download_gtsrb_once(raw_dir: Path) -> None:
    print("[Dataset] Checking GTSRB in cache...")
    raw_gtsrb_dir = raw_dir / "gtsrb"
    train_ok = any(raw_gtsrb_dir.rglob("Training"))
    test_ok = any(raw_gtsrb_dir.rglob("GT-final_test.csv"))

    if train_ok and test_ok:
        print("[Dataset] Raw dataset already exists. Download skipped.")
        return

    print("[Dataset] Downloading GTSRB (train + test) from open source mirror...")
    GTSRB(root=str(raw_dir), split="train", download=True)
    GTSRB(root=str(raw_dir), split="test", download=True)
    print("[Dataset] Download complete.")


def _read_train_samples(train_dir: Path) -> list[tuple[Path, int]]:
    rows: list[tuple[Path, int]] = []
    for class_dir in sorted(train_dir.iterdir()):
        if not class_dir.is_dir() or not class_dir.name.isdigit():
            continue
        label = int(class_dir.name)
        for img_path in class_dir.rglob("*"):
            if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append((img_path, label))
    return rows


def _read_test_samples(test_dir: Path, test_csv: Path) -> list[tuple[Path, int]]:
    rows: list[tuple[Path, int]] = []
    with test_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            name = row["Filename"]
            label = int(row["ClassId"])
            img_path = test_dir / name
            if img_path.exists():
                rows.append((img_path, label))
    return rows


def _split_train_val(
    train_samples: list[tuple[Path, int]],
    val_ratio: float,
    seed: int,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    by_class: dict[int, list[Path]] = defaultdict(list)
    for p, label in train_samples:
        by_class[label].append(p)

    rng = random.Random(seed)
    train_rows: list[tuple[Path, int]] = []
    val_rows: list[tuple[Path, int]] = []

    for label, paths in by_class.items():
        paths = paths.copy()
        rng.shuffle(paths)

        if len(paths) <= 2:
            split_idx = max(1, len(paths) - 1)
        else:
            val_count = max(1, int(round(len(paths) * val_ratio)))
            val_count = min(val_count, len(paths) - 1)
            split_idx = len(paths) - val_count

        train_rows.extend((p, label) for p in paths[:split_idx])
        val_rows.extend((p, label) for p in paths[split_idx:])

    return train_rows, val_rows


def _split_test_demo(
    test_samples: list[tuple[Path, int]],
    demo_samples_per_class: int,
    seed: int,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    by_class: dict[int, list[Path]] = defaultdict(list)
    for p, label in test_samples:
        by_class[label].append(p)

    rng = random.Random(seed + 17)
    test_rows: list[tuple[Path, int]] = []
    demo_rows: list[tuple[Path, int]] = []

    for label, paths in by_class.items():
        paths = paths.copy()
        rng.shuffle(paths)
        demo_count = min(demo_samples_per_class, len(paths))

        demo_rows.extend((p, label) for p in paths[:demo_count])
        test_rows.extend((p, label) for p in paths[demo_count:])

    return test_rows, demo_rows


def _write_demo_images(demo_rows: list[tuple[Path, int]], demo_root: Path) -> list[tuple[Path, int]]:
    if demo_root.exists():
        shutil.rmtree(demo_root)
    demo_root.mkdir(parents=True, exist_ok=True)

    output_rows: list[tuple[Path, int]] = []
    for idx, (src_path, label) in enumerate(tqdm(demo_rows, desc="[Dataset] Writing demo images"), start=1):
        class_dir = demo_root / f"{label:02d}"
        class_dir.mkdir(parents=True, exist_ok=True)

        dst_name = f"{idx:06d}_{src_path.stem}.png"
        dst_path = class_dir / dst_name

        image = Image.open(src_path).convert("RGB")
        image.save(dst_path)
        output_rows.append((dst_path, label))

    return output_rows


def _write_reference_icons(train_rows: list[tuple[Path, int]], icon_root: Path) -> None:
    if icon_root.exists():
        for old in icon_root.glob("*.png"):
            old.unlink(missing_ok=True)
    icon_root.mkdir(parents=True, exist_ok=True)

    chosen: dict[int, Path] = {}
    for p, label in train_rows:
        if label not in chosen:
            chosen[label] = p

    for label in sorted(chosen):
        image = Image.open(chosen[label]).convert("RGB").resize((96, 96))
        image.save(icon_root / f"{label:02d}.png")


def _generate_preview_image(manifest_df: pd.DataFrame, output_path: Path, seed: int = 42) -> None:
    rng = random.Random(seed + 99)
    splits = ["train", "val", "test", "demo"]
    rows: list[tuple[str, Path, int]] = []

    for split in splits:
        part = manifest_df[manifest_df["split"] == split]
        if part.empty:
            continue

        pick_count = min(8, len(part))
        picks = part.sample(n=pick_count, random_state=seed + len(split))
        for _, rec in picks.iterrows():
            rows.append((split, Path(rec["path"]), int(rec["label"])))

    if not rows:
        return

    tile = 96
    cols = 8
    header_h = 24
    padding = 8
    rows_count = (len(rows) + cols - 1) // cols

    canvas_w = padding + cols * (tile + padding)
    canvas_h = padding + rows_count * (tile + header_h + padding)
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(245, 247, 250))
    draw = ImageDraw.Draw(canvas)

    for idx, (split, img_path, label) in enumerate(rows):
        r = idx // cols
        c = idx % cols
        x = padding + c * (tile + padding)
        y = padding + r * (tile + header_h + padding)

        try:
            patch = Image.open(img_path).convert("RGB").resize((tile, tile))
            canvas.paste(patch, (x, y + header_h))
            draw.rectangle((x, y + header_h, x + tile, y + header_h + tile), outline=(90, 90, 90), width=1)
            draw.text((x + 2, y + 2), f"{split} | c={label}", fill=(30, 30, 30))
        except Exception:
            draw.rectangle((x, y + header_h, x + tile, y + header_h + tile), outline=(200, 80, 80), width=2)
            draw.text((x + 2, y + 2), f"{split} | c={label}", fill=(200, 80, 80))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def inspect_prepared_dataset() -> dict:
    ensure_project_dirs()
    if not MANIFEST_PATH.exists() or not DATASET_META_PATH.exists():
        raise FileNotFoundError("Prepared dataset not found. Run --mode prepare first.")

    df = pd.read_csv(MANIFEST_PATH)
    split_counts = {k: int(v) for k, v in df.groupby("split").size().to_dict().items()}

    class_counts: dict[str, dict[str, int]] = {}
    for split in sorted(df["split"].unique()):
        part = df[df["split"] == split]
        counts = part.groupby("label").size().to_dict()
        class_counts[split] = {str(int(k)): int(v) for k, v in counts.items()}

    summary = {
        "manifest_path": str(MANIFEST_PATH),
        "total_samples": int(len(df)),
        "split_counts": split_counts,
        "preview_image": str(DATASET_PREVIEW_PATH) if DATASET_PREVIEW_PATH.exists() else None,
        "class_counts": class_counts,
    }

    print("[Dataset] Summary:")
    print(summary)
    return summary


def prepare_dataset(force: bool = False, settings: DatasetSettings | None = None) -> dict:
    ensure_project_dirs()
    settings = settings or DatasetSettings()

    if MANIFEST_PATH.exists() and DATASET_META_PATH.exists() and not force:
        print("[Dataset] Prepared dataset already exists. Skipping rebuild.")
        return inspect_prepared_dataset()

    if force and PROCESSED_DATA_DIR.exists():
        shutil.rmtree(PROCESSED_DATA_DIR)
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    _download_gtsrb_once(RAW_DATA_DIR)
    train_dir, test_dir, test_csv = _locate_raw_paths(RAW_DATA_DIR)

    print("[Dataset] Scanning raw files...")
    train_samples = _read_train_samples(train_dir)
    test_samples = _read_test_samples(test_dir, test_csv)

    if not train_samples or not test_samples:
        raise RuntimeError("Raw dataset is empty after scan. Check download integrity.")

    train_rows, val_rows = _split_train_val(
        train_samples=train_samples,
        val_ratio=settings.val_ratio,
        seed=settings.seed,
    )
    test_rows, demo_rows = _split_test_demo(
        test_samples=test_samples,
        demo_samples_per_class=settings.demo_samples_per_class,
        seed=settings.seed,
    )

    demo_root = PROCESSED_DATA_DIR / "demo"
    demo_rows_written = _write_demo_images(demo_rows, demo_root)
    _write_reference_icons(train_rows, REFERENCE_ICONS_DIR)

    manifest_rows = []
    for split_name, rows in (
        ("train", train_rows),
        ("val", val_rows),
        ("test", test_rows),
    ):
        for p, label in rows:
            manifest_rows.append({"split": split_name, "path": str(p.resolve()), "label": int(label)})

    for p, label in demo_rows_written:
        manifest_rows.append({"split": "demo", "path": str(p.resolve()), "label": int(label)})

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(MANIFEST_PATH, index=False)

    class_map = {str(k): v for k, v in GTSRB_CLASS_NAMES_RU.items()}
    dump_json(CLASS_MAP_PATH, class_map)

    _generate_preview_image(manifest_df, DATASET_PREVIEW_PATH, seed=settings.seed)

    split_counts = {str(k): int(v) for k, v in manifest_df.groupby("split").size().to_dict().items()}
    per_split_class_counts: dict[str, dict[str, int]] = {}
    for split in sorted(manifest_df["split"].unique()):
        counts = manifest_df[manifest_df["split"] == split].groupby("label").size().to_dict()
        per_split_class_counts[split] = {str(int(k)): int(v) for k, v in counts.items()}

    meta = {
        "created_at": now_iso(),
        "dataset": "GTSRB (open)",
        "source": {
            "train_dir": str(train_dir),
            "test_dir": str(test_dir),
            "test_csv": str(test_csv),
        },
        "settings": asdict(settings),
        "split_counts": split_counts,
        "class_counts": per_split_class_counts,
        "preview_image": str(DATASET_PREVIEW_PATH),
        "manifest_path": str(MANIFEST_PATH),
    }
    dump_json(DATASET_META_PATH, meta)

    print("[Dataset] Done.")
    print(f"[Dataset] Split counts: {split_counts}")
    print(f"[Dataset] Manifest: {MANIFEST_PATH}")
    print(f"[Dataset] Preview:  {DATASET_PREVIEW_PATH}")

    return meta
