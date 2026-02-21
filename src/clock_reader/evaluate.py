"""Compare reading methods on a held-out synthetic set and, optionally, labelled real photos.

    clock-eval --checkpoint runs/resnet34_unet/best.pt --n 2000
    clock-eval --checkpoint best.pt --real-csv data/real_clocks/labels.csv

Methods: ``network`` (learned segmentation + geometry), ``classical`` (Hough + threshold +
geometry) and, on synthetic data, ``oracle`` (ground-truth masks + geometry, an upper bound
for the geometric step).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from clock_reader.classical import classical_mask
from clock_reader.config import TrainConfig
from clock_reader.geometry import circular_minutes_error, read_time
from clock_reader.inference import ClockReader
from clock_reader.metrics import reading_metrics
from clock_reader.synth import ClockRenderer, SynthConfig
from clock_reader.utils.config import load_config

TEST_SEED = 5_000_000  # far from the training and validation seeds


def score(masks: list[np.ndarray], targets: list[float]) -> dict:
    errors, failures = [], 0
    for mask, target in zip(masks, targets, strict=True):
        reading = read_time(mask)
        if reading is None:
            failures += 1
        else:
            errors.append(circular_minutes_error(reading.minutes_float, target))
    return {k: round(v, 4) for k, v in reading_metrics(errors, failures).items()}


def evaluate(
    reader: ClockReader | None,
    images: list[np.ndarray],
    targets: list[float],
    gt_masks: list[np.ndarray] | None = None,
    batch: int = 32,
) -> dict[str, dict]:
    results = {"classical": score([classical_mask(im) for im in images], targets)}
    if reader is not None:
        masks = [m for i in range(0, len(images), batch) for m in reader.segment(images[i : i + batch])]
        results["network"] = score(masks, targets)
    if gt_masks is not None:
        results["oracle"] = score(gt_masks, targets)
    return results


def print_table(title: str, results: dict[str, dict]) -> None:
    print(f"\n{title}")
    print("| method | acc@1 min | acc@5 min | acc@15 min | median err | fail rate |")
    print("|---|---|---|---|---|---|")
    for name, m in results.items():
        print(
            f"| {name} | {m['acc_1min']:.3f} | {m['acc_5min']:.3f} | {m['acc_15min']:.3f} | "
            f"{m['median_error_min']:.1f} min | {m['fail_rate']:.3f} |"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, help="omit to evaluate only the classical baseline")
    parser.add_argument("--n", type=int, default=2000, help="synthetic test clocks")
    parser.add_argument("--real-csv", type=Path, help="CSV with path,hour,minute[,second]")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, help="default: <checkpoint dir>/eval.json")
    parser.add_argument(
        "--test-config",
        type=Path,
        help="training config whose synth section defines the test clocks (default: the checkpoint's "
        "own). Ablations need the main config here, or they are tested on their own easier clocks.",
    )
    args = parser.parse_args(argv)

    reader = ClockReader(args.checkpoint, args.device) if args.checkpoint else None
    if args.test_config:
        synth = load_config(TrainConfig, args.test_config).synth
    else:
        synth = reader.cfg.synth if reader else SynthConfig()
    flat = dataclasses.replace(synth, max_tilt=0.0, max_roll_deg=0.0)  # facing the camera, upright
    report = {}
    for name, cfg in (("synthetic", synth), ("synthetic_flat", flat)):
        renderer = ClockRenderer(cfg)
        images, gt_masks, targets = [], [], []
        for i in range(args.n):
            image, mask, t = renderer.render(TEST_SEED + i)
            images.append(image)
            gt_masks.append(mask)
            targets.append(t.minutes)
        report[name] = evaluate(reader, images, targets, gt_masks)
        print_table(f"{name} ({args.n} clocks)", report[name])

    if args.real_csv:
        table = pd.read_csv(args.real_csv)
        real_images = [
            cv2.cvtColor(cv2.imread(str(args.real_csv.parent / p)), cv2.COLOR_BGR2RGB) for p in table["path"]
        ]
        seconds = table["second"].fillna(0) if "second" in table else pd.Series(0, index=table.index)
        real_targets = [
            (h % 12) * 60 + m + s / 60
            for h, m, s in zip(table["hour"], table["minute"], seconds, strict=True)
        ]
        report["real"] = evaluate(reader, real_images, real_targets)
        print_table(f"Real photos ({len(table)})", report["real"])

    out = args.out or ((args.checkpoint.parent if args.checkpoint else Path(".")) / "eval.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
