from __future__ import annotations

from PIL import Image
from torchvision import transforms

from .config import NORMALIZATION_MEAN, NORMALIZATION_STD


def build_train_transform(input_size: int):
    return transforms.Compose(
        [
            transforms.Resize((input_size + 12, input_size + 12)),
            transforms.RandomCrop((input_size, input_size)),
            transforms.RandomAffine(
                degrees=12,
                translate=(0.12, 0.12),  # includes shift augmentation
                scale=(0.90, 1.10),
                shear=8,
                fill=0,
            ),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD),
        ]
    )


def build_eval_transform(input_size: int):
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD),
        ]
    )


def make_model_input_preview(image: Image.Image, input_size: int) -> Image.Image:
    return image.convert("RGB").resize((input_size, input_size))
