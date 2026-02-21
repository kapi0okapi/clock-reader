"""U-Net style segmentation network: pretrained timm encoder + light decoder with skip connections."""

from __future__ import annotations

import timm
import torch
import torch.nn.functional as F
from torch import nn


def conv_bn_relu(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)
    )


class DecoderBlock(nn.Module):
    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.block = nn.Sequential(conv_bn_relu(cin + cskip, cout), conv_bn_relu(cout, cout))

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        size = skip.shape[-2:] if skip is not None else [2 * s for s in x.shape[-2:]]
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.block(x)


class ClockSegNet(nn.Module):
    def __init__(
        self,
        encoder: str = "resnet34",
        num_classes: int = 7,
        pretrained: bool = True,
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16),
    ):
        super().__init__()
        self.encoder = timm.create_model(encoder, pretrained=pretrained, features_only=True)
        skips = self.encoder.feature_info.channels()  # strides 2, 4, 8, 16, 32
        cin = skips[-1]
        blocks = []
        for i, cout in enumerate(decoder_channels):
            skip = skips[-2 - i] if i + 1 < len(skips) else 0
            blocks.append(DecoderBlock(cin, skip, cout))
            cin = cout
        self.decoder = nn.ModuleList(blocks)
        self.head = nn.Conv2d(cin, num_classes, 1)

    @property
    def mean_std(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        cfg = self.encoder.pretrained_cfg
        return tuple(cfg["mean"]), tuple(cfg["std"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        y = features[-1]
        skips = features[-2::-1]
        for i, block in enumerate(self.decoder):
            y = block(y, skips[i] if i < len(skips) else None)
        y = self.head(y)
        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return y


class SegLoss(nn.Module):
    """Weighted cross-entropy + soft Dice over the small classes (hands, pivot, top), which cover
    only a few percent of the pixels."""

    def __init__(
        self,
        class_weights: list[float],
        dice_classes: tuple[int, ...] = (2, 3, 4, 5, 6),
        dice_weight: float = 0.5,
    ):
        super().__init__()
        self.register_buffer("weights", torch.tensor(class_weights, dtype=torch.float32))
        self.dice_classes = list(dice_classes)
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        ce = F.cross_entropy(logits, target, weight=self.weights)
        probs = logits.softmax(1)[:, self.dice_classes]
        onehot = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).float()[:, self.dice_classes]
        inter = (probs * onehot).sum((0, 2, 3))
        union = probs.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
        dice = 1 - ((2 * inter + 1) / (union + 1)).mean()
        return ce + self.dice_weight * dice
