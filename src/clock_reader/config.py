"""Experiment configuration. YAML files in ``configs/`` override these defaults."""

from __future__ import annotations

from dataclasses import dataclass, field

from clock_reader.synth import SynthConfig


@dataclass
class DataConfig:
    train_size: int = 40_000  # fresh synthetic samples per epoch
    val_size: int = 2_000
    train_seed: int = 1_000_000
    val_seed: int = 0
    batch_size: int = 32
    num_workers: int = 16  # rendering is CPU-bound; keep the GPU fed
    real_csv: str = ""  # optional labelled real photos, evaluated after training


@dataclass
class ModelConfig:
    encoder: str = "resnet34"
    pretrained: bool = True
    class_weights: tuple[float, ...] = (0.5, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0)
    dice_weight: float = 0.5


@dataclass
class OptimConfig:
    epochs: int = 40
    lr: float = 1e-3
    encoder_lr_mult: float = 0.3
    weight_decay: float = 1e-4
    warmup_epochs: float = 1.0
    max_grad_norm: float = 5.0


@dataclass
class TrainConfig:
    name: str = "resnet34_unet"
    seed: int = 0
    device: str = "auto"
    amp: bool = True
    synth: SynthConfig = field(default_factory=SynthConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
