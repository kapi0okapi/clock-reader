"""Train the clock segmentation network on synthetic data.

    clock-train --config configs/resnet34_unet.yaml
    clock-train --config configs/smoke.yaml --run-dir /tmp/smoke

Model selection uses the end-to-end metric (share of validation clocks read within 5 minutes),
not pixel IoU: a hand mask can look fine and still put the tip a few degrees off.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from clock_reader.config import TrainConfig
from clock_reader.data import LabelledClocks, SyntheticClocks
from clock_reader.geometry import circular_minutes_error, read_time
from clock_reader.metrics import confusion_matrix, iou_per_class, reading_metrics
from clock_reader.model import ClockSegNet, SegLoss
from clock_reader.synth import CLASS_NAMES
from clock_reader.utils.config import load_config, save_config
from clock_reader.utils.runtime import (
    MetricLogger,
    Precision,
    StopSignal,
    cosine_with_warmup,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    select_device,
    setup_logging,
)

log = logging.getLogger("clock_reader.train")


def build_model(cfg: TrainConfig, pretrained: bool | None = None) -> ClockSegNet:
    pretrained = cfg.model.pretrained if pretrained is None else pretrained
    return ClockSegNet(cfg.model.encoder, num_classes=len(CLASS_NAMES), pretrained=pretrained)


def loader(dataset, cfg: TrainConfig, shuffle: bool, persistent: bool = False) -> DataLoader:
    # The training loader is rebuilt every epoch so that workers pick up the new epoch seed.
    return DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent and cfg.data.num_workers > 0,
    )


@torch.no_grad()
def evaluate_reading(
    model: ClockSegNet, data: DataLoader, device: torch.device, precision: Precision
) -> dict:
    """Segment, read the time from each predicted mask, and compare with the label."""
    model.eval()
    num_classes = len(CLASS_NAMES)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    errors, failures = [], 0
    for images, masks, minutes in data:
        with precision.autocast():
            logits = model(images.to(device, non_blocking=True))
        pred = logits.argmax(1).cpu().numpy().astype(np.uint8)
        if masks.ndim == 3:  # synthetic data has ground-truth masks
            cm += confusion_matrix(pred, masks.numpy(), num_classes)
        for p, target in zip(pred, minutes.tolist(), strict=True):
            reading = read_time(p)
            if reading is None:
                failures += 1
            else:
                errors.append(circular_minutes_error(reading.minutes_float, target))
    metrics = reading_metrics(errors, failures)
    if cm.sum():
        iou = iou_per_class(cm)
        metrics["miou"] = float(np.nanmean(iou))
        metrics.update({f"iou_{name}": float(v) for name, v in zip(CLASS_NAMES, iou, strict=True)})
    return metrics


def train(cfg: TrainConfig, run_dir: Path, resume: bool = True) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    seed_everything(cfg.seed)
    device = select_device(cfg.device)
    precision = Precision(device, cfg.amp)
    log.info("device=%s precision=%s run_dir=%s", device, precision.dtype, run_dir)

    model = build_model(cfg).to(device)
    mean, std = model.mean_std
    train_set = SyntheticClocks(cfg.data.train_size, cfg.data.train_seed, mean, std, cfg.synth)
    val_set = SyntheticClocks(cfg.data.val_size, cfg.data.val_seed, mean, std, cfg.synth)
    val_loader = loader(val_set, cfg, shuffle=False, persistent=True)

    o = cfg.optim
    encoder = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [{"params": encoder, "lr": o.lr * o.encoder_lr_mult}, {"params": rest, "lr": o.lr}],
        weight_decay=o.weight_decay,
    )
    steps = cfg.data.train_size // cfg.data.batch_size
    scheduler = cosine_with_warmup(optimizer, int(o.warmup_epochs * steps), o.epochs * steps)
    criterion = SegLoss(list(cfg.model.class_weights), dice_weight=cfg.model.dice_weight).to(device)
    logger = MetricLogger(run_dir)
    stop = StopSignal()

    start_epoch, best = 0, {"acc_5min": -1.0}
    last_path = run_dir / "last.pt"
    if resume and last_path.exists():
        state = load_checkpoint(last_path, device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        precision.scaler.load_state_dict(state["scaler"])
        start_epoch, best = state["epoch"], state["best"]
        log.info("resumed at epoch %d", start_epoch)

    def save_last(next_epoch: int) -> None:
        save_checkpoint(
            last_path,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            scaler=precision.scaler.state_dict(),
            epoch=next_epoch,
            best=best,
            config=dataclasses.asdict(cfg),
        )

    for epoch in range(start_epoch, o.epochs):
        train_set.set_epoch(epoch)  # new synthetic samples every epoch
        model.train()
        started, total = time.time(), 0.0
        for images, masks, _ in loader(train_set, cfg, shuffle=True):
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            with precision.autocast():
                logits = model(images)
            loss = criterion(logits, masks)
            precision.step(loss, optimizer, model, o.max_grad_norm)
            scheduler.step()
            total += loss.item()
            if stop:
                save_last(epoch)
                log.warning("stopped by signal during epoch %d, checkpoint saved", epoch)
                return best

        val = evaluate_reading(model, val_loader, device, precision)
        logger.log(
            epoch,
            {
                "train/loss": total / steps,
                **{f"val/{k}": v for k, v in val.items()},
                "lr": scheduler.get_last_lr()[-1],
            },
        )
        log.info(
            "epoch %d/%d loss=%.4f mIoU=%.3f acc@5min=%.3f mean_err=%.1fmin (%.0fs)",
            epoch + 1,
            o.epochs,
            total / steps,
            val["miou"],
            val["acc_5min"],
            val["mean_error_min"],
            time.time() - started,
        )
        if val["acc_5min"] > best["acc_5min"]:
            best = val
            save_checkpoint(
                run_dir / "best.pt",
                model=model.state_dict(),
                config=dataclasses.asdict(cfg),
                epoch=epoch,
                val=val,
            )
        save_last(epoch + 1)

    if cfg.data.real_csv:
        state = load_checkpoint(run_dir / "best.pt", device)
        model.load_state_dict(state["model"])
        real = LabelledClocks(os.path.expandvars(cfg.data.real_csv), cfg.synth.size, mean, std)
        log.info(
            "real photos: %s", evaluate_reading(model, loader(real, cfg, shuffle=False), device, precision)
        )
    logger.close()
    return best


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--run-dir", type=str, help="default: $RUNS/<name> (or runs/<name>)")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    setup_logging()
    cfg = load_config(TrainConfig, args.config, args.set)
    run_dir = Path(os.path.expandvars(args.run_dir or os.path.join(os.environ.get("RUNS", "runs"), cfg.name)))
    train(cfg, run_dir, resume=not args.no_resume)


if __name__ == "__main__":
    main()
