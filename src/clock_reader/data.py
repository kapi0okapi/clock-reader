"""Datasets: synthetic clocks rendered on the fly, and labelled real photos for evaluation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from clock_reader.synth import ClockRenderer, SynthConfig


def to_tensor(rgb: np.ndarray, mean, std) -> torch.Tensor:
    x = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255)
    return (x - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]


class SyntheticClocks(Dataset):
    """``length`` clocks rendered from seeds ``seed + epoch * length + index``. Calling
    ``set_epoch`` gives the training set new samples every epoch; validation keeps epoch 0."""

    def __init__(self, length: int, seed: int, mean, std, synth: SynthConfig | None = None):
        self.length = length
        self.seed = seed
        self.epoch = 0
        self.mean, self.std = mean, std
        self.renderer = ClockRenderer(synth)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int):
        image, mask, t = self.renderer.render(self.seed + self.epoch * self.length + i)
        return to_tensor(image, self.mean, self.std), torch.from_numpy(mask).long(), torch.tensor(t.minutes)


class LabelledClocks(Dataset):
    """Real photos of clocks listed in a CSV with columns ``path, hour, minute`` (optional ``second``).
    Paths are relative to the CSV. Images are resized to a square; the rectification step in
    the geometry absorbs the resulting affine distortion."""

    def __init__(self, csv_path: str | Path, size: int, mean, std):
        self.csv_path = Path(csv_path)
        self.table = pd.read_csv(self.csv_path)
        self.size, self.mean, self.std = size, mean, std

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, i: int):
        row = self.table.iloc[i]
        bgr = cv2.imread(str(self.csv_path.parent / row["path"]))
        if bgr is None:
            raise OSError(f"cannot read {row['path']}")
        rgb = cv2.cvtColor(
            cv2.resize(bgr, (self.size, self.size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB
        )
        second = row.get("second", 0)
        minutes = (
            (int(row["hour"]) % 12) * 60 + int(row["minute"]) + (0 if pd.isna(second) else float(second)) / 60
        )
        return to_tensor(rgb, self.mean, self.std), torch.zeros(1), torch.tensor(minutes)
