"""Read the time from photos.

    clock-read --checkpoint best.pt --source photo.jpg            # the photo is a clock crop
    clock-read --checkpoint best.pt --source photos/ --detect     # find clocks first (YOLO, COCO "clock")

Writes an annotated copy of each image to --out: dial ellipse, pivot, hand directions, time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from clock_reader.geometry import Reading, hand_tip
from clock_reader.inference import ClockReader

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COCO_CLOCK = 74
OVERLAY = np.array(
    [[0, 0, 0], [90, 90, 90], [230, 60, 60], [60, 200, 60], [60, 60, 230], [240, 240, 0], [240, 0, 240]],
    np.uint8,
)


def detect_clocks(rgb: np.ndarray, detector, margin: float = 0.12) -> list[tuple[int, int, int, int]]:
    result = detector.predict(rgb[..., ::-1], classes=[COCO_CLOCK], conf=0.25, nms=False, verbose=False)[0]
    h, w = rgb.shape[:2]
    boxes = []
    for x0, y0, x1, y1 in result.boxes.xyxy.cpu().numpy():
        mx, my = (x1 - x0) * margin, (y1 - y0) * margin
        boxes.append((max(0, int(x0 - mx)), max(0, int(y0 - my)), min(w, int(x1 + mx)), min(h, int(y1 + my))))
    return boxes


def draw(bgr: np.ndarray, box, reading: Reading | None, mask: np.ndarray) -> None:
    x0, y0, x1, y1 = box
    crop = bgr[y0:y1, x0:x1]
    tint = OVERLAY[mask][..., ::-1]
    crop[mask > 0] = (0.6 * crop[mask > 0] + 0.4 * tint[mask > 0]).astype(np.uint8)
    cv2.rectangle(bgr, (x0, y0), (x1, y1), (0, 220, 255), 2)
    if reading is None:
        cv2.putText(bgr, "?", (x0 + 4, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return
    (cx, cy), axes, angle = reading.ellipse
    cv2.ellipse(bgr, ((cx + x0, cy + y0), axes, angle), (255, 255, 255), 1, cv2.LINE_AA)
    offset = np.array([x0, y0])
    pivot = (np.array(reading.pivot) + offset).astype(int)
    for name, length, color in (("hour", 0.5, (60, 60, 230)), ("minute", 0.85, (60, 200, 60))):
        tip = (hand_tip(reading, name, length) + offset).astype(int)
        cv2.arrowedLine(bgr, tuple(pivot), tuple(tip), color, 2, cv2.LINE_AA)
    label = f"{reading} ({reading.confidence:.2f})"
    cv2.putText(bgr, label, (x0 + 4, max(18, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("predictions"))
    parser.add_argument("--detect", action="store_true", help="locate clocks with a COCO YOLO model first")
    parser.add_argument(
        "--detector", default="models/yolo26s.pt", help="COCO YOLO weights (NMS-free YOLO26 by default)"
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    reader = ClockReader(args.checkpoint, args.device)
    detector = None
    if args.detect:
        from ultralytics import YOLO

        detector = YOLO(args.detector)
    paths = (
        sorted(p for p in args.source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if args.source.is_dir()
        else [args.source]
    )
    args.out.mkdir(parents=True, exist_ok=True)

    for path in paths:
        bgr = cv2.imread(str(path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        boxes = detect_clocks(rgb, detector) if detector else [(0, 0, rgb.shape[1], rgb.shape[0])]
        readings = []
        for box in boxes:
            x0, y0, x1, y1 = box
            reading, mask = reader.read(rgb[y0:y1, x0:x1])
            draw(bgr, box, reading, mask)
            readings.append(str(reading) if reading else "unreadable")
        cv2.imwrite(str(args.out / path.name), bgr)
        print(f"{path.name}: {', '.join(readings) or 'no clock found'}")


if __name__ == "__main__":
    main()
