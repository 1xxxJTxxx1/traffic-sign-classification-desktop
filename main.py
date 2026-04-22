from __future__ import annotations

import argparse
import sys

from src.config import MANIFEST_PATH, DatasetSettings, TrainingSettings


RETRAIN_REASON_TEXT = {
    "weights_missing": "лучшие веса отсутствуют",
    "weights_unreadable": "файл весов поврежден или не читается",
    "dataset_manifest_missing": "отсутствует подготовленный датасет",
    "checkpoint_without_dataset_fingerprint": "весы старого формата (без отпечатка датасета)",
    "dataset_manifest_changed": "датасет изменился",
    "dataset_class_map_changed": "изменилось соответствие классов",
    "checkpoint_without_model_signature": "весы старого формата (без сигнатуры архитектуры)",
    "architecture_changed": "архитектура модели изменилась",
    "num_classes_changed": "изменилось количество классов",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Traffic Sign Classification Coursework")
    parser.add_argument(
        "--mode",
        choices=["auto", "prepare", "inspect", "train", "app"],
        default="auto",
        help="auto (default): smart startup; prepare: download+split dataset; inspect: show dataset summary; train: train CNN; app: run Tkinter UI",
    )

    parser.add_argument("--force-dataset", action="store_true", help="Rebuild prepared dataset even if it already exists")
    parser.add_argument("--force-retrain", action="store_true", help="Force model retraining")

    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio from official train split")
    parser.add_argument("--demo-per-class", type=int, default=25, help="How many demo images per class")

    parser.add_argument("--epochs", type=int, default=35, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=96, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 is safest on Windows)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--require-gpu", action="store_true", help="Fail training if CUDA is unavailable")

    return parser


def _build_dataset_settings(args: argparse.Namespace) -> DatasetSettings:
    return DatasetSettings(
        val_ratio=args.val_ratio,
        demo_samples_per_class=args.demo_per_class,
        seed=42,
    )


def _build_training_settings(args: argparse.Namespace) -> TrainingSettings:
    return TrainingSettings(
        input_size=64,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=1e-4,
        label_smoothing=0.05,
        num_workers=args.num_workers,
        early_stopping_patience=7,
        require_gpu=args.require_gpu,
    )


def run_prepare(args: argparse.Namespace) -> None:
    from src.prepare_dataset import prepare_dataset

    settings = _build_dataset_settings(args)
    prepare_dataset(force=args.force_dataset, settings=settings)


def run_inspect() -> None:
    from src.prepare_dataset import inspect_prepared_dataset

    try:
        inspect_prepared_dataset()
    except FileNotFoundError as exc:
        print(f"[Main] {exc}")
        print("[Main] Сначала запустите: python main.py --mode prepare")


def run_train(args: argparse.Namespace, force_retrain_override: bool | None = None) -> dict:
    from src.train import train_model

    if not MANIFEST_PATH.exists():
        print("[Main] Prepared dataset not found. Running dataset preparation first...")
        run_prepare(args)

    settings = _build_training_settings(args)
    force_retrain = args.force_retrain if force_retrain_override is None else force_retrain_override

    return train_model(settings=settings, force_retrain=force_retrain, seed=42)


def run_app() -> None:
    from src.gui import launch_app

    try:
        launch_app()
    except FileNotFoundError as exc:
        print(f"[Main] {exc}")
        print("[Main] Сначала обучите модель: python main.py --mode train")


def _print_retrain_reasons(reasons: list[str]) -> None:
    for reason in reasons:
        text = RETRAIN_REASON_TEXT.get(reason, reason)
        print(f"[Auto] Причина переобучения: {text}")


def run_auto(args: argparse.Namespace) -> None:
    print("[Auto] Smart start mode")
    from src.train import inspect_weights_compatibility

    if not MANIFEST_PATH.exists() or args.force_dataset:
        print("[Auto] Подготовленный датасет отсутствует или запрошена пересборка. Запускаю prepare...")
        run_prepare(args)

    settings = _build_training_settings(args)
    compatibility = inspect_weights_compatibility(settings)

    if args.force_retrain:
        print("[Auto] Включен --force-retrain. Запускаю обучение.")
        run_train(args, force_retrain_override=True)
        run_app()
        return

    if compatibility["needs_retrain"]:
        print("[Auto] Обнаружено, что текущие веса неактуальны.")
        _print_retrain_reasons(compatibility["reasons"])
        run_train(args, force_retrain_override=True)
        run_app()
        return

    print("[Auto] Веса актуальны. Открываю приложение без переобучения.")
    run_app()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "auto":
        run_auto(args)
    elif args.mode == "prepare":
        run_prepare(args)
    elif args.mode == "inspect":
        run_inspect()
    elif args.mode == "train":
        run_train(args)
    elif args.mode == "app":
        run_app()
    else:
        parser.error(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
