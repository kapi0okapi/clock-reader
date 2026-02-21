"""Trained-model wrapper: RGB image -> segmentation mask -> time reading."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from clock_reader.config import TrainConfig
from clock_reader.data import to_tensor
from clock_reader.geometry import Reading, read_time
from clock_reader.model import ClockSegNet
from clock_reader.synth import CLASS_NAMES
from clock_reader.utils.config import from_dict
from clock_reader.utils.runtime import load_checkpoint


class ClockReader:
    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        self.device = torch.device(device)
        state = load_checkpoint(Path(checkpoint), self.device)
        self.cfg = from_dict(TrainConfig, state["config"])
        self.model = ClockSegNet(self.cfg.model.encoder, num_classes=len(CLASS_NAMES), pretrained=False)
        self.model.load_state_dict(state["model"])
        self.model.to(self.device).eval()
        self.size = self.cfg.synth.size
        self.mean, self.std = self.model.mean_std

    @torch.no_grad()
    def segment(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Masks at each input's own resolution."""
        batch = [cv2.resize(im, (self.size, self.size), interpolation=cv2.INTER_AREA) for im in images]
        x = torch.stack([to_tensor(im, self.mean, self.std) for im in batch]).to(self.device)
        pred = self.model(x).argmax(1).cpu().numpy().astype(np.uint8)
        return [
            cv2.resize(p, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
            for p, im in zip(pred, images, strict=True)
        ]

    def read(self, rgb: np.ndarray) -> tuple[Reading | None, np.ndarray]:
        mask = self.segment([rgb])[0]
        return read_time(mask), mask
