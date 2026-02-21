"""Classical baseline without any learning: Hough circle for the dial, Otsu threshold for the hands.

It produces a coarse mask (dial + unlabelled hand pixels) that goes through the same geometry
as the network output. Comparing the two shows what the learned segmentation adds: robustness
to tilt, clutter, low contrast and decorative dials.
"""

from __future__ import annotations

import cv2
import numpy as np

from clock_reader.synth import DIAL, HOUR


def classical_mask(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=w,
        param1=100,
        param2=30,
        minRadius=int(0.15 * min(h, w)),
        maxRadius=int(0.55 * min(h, w)),
    )
    mask = np.zeros((h, w), np.uint8)
    if circles is None:
        return mask
    cx, cy, r = circles[0, 0]
    cv2.circle(mask, (int(cx), int(cy)), int(r), DIAL, -1)

    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.hypot(xx - cx, yy - cy)
    inside = dist < 0.85 * r  # ignore the rim and the outer tick ring
    values = blurred[inside]
    _, thresh = cv2.threshold(values.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = values[thresh.ravel() > 0].min() if (thresh > 0).any() else 127
    dark_hands = np.median(values) > threshold  # hands contrast with the dial in either direction
    ink = (blurred < threshold) if dark_hands else (blurred >= threshold)
    ink = (ink & inside).astype(np.uint8)

    # keep only components that reach the centre: hands, not numerals or ticks
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
    hands = np.zeros_like(ink)
    for i in range(1, n):
        component = labels == i
        if dist[component].min() < 0.12 * r and stats[i, cv2.CC_STAT_AREA] > 10:
            hands |= component.astype(np.uint8)
    mask[hands > 0] = HOUR  # unlabelled: geometry splits them by direction and length
    return mask
