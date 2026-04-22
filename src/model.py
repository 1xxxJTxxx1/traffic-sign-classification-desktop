from __future__ import annotations

import hashlib
import inspect
import torch
from torch import nn

MODEL_ARCH_VERSION = "traffic_sign_cnn_v1"


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TrafficSignCNN(nn.Module):
    """Self-written CNN for traffic sign classification."""

    def __init__(self, num_classes: int = 43) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32, dropout=0.05),
            ConvBlock(32, 64, dropout=0.08),
            ConvBlock(64, 128, dropout=0.10),
            ConvBlock(128, 256, dropout=0.12),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.35),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_signature() -> str:
    try:
        source = inspect.getsource(ConvBlock) + "\n" + inspect.getsource(TrafficSignCNN)
    except OSError:
        source = "source_unavailable"
    payload = f"{MODEL_ARCH_VERSION}\n{source}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
