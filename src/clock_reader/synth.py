"""Procedural analogue clock renderer.

Every sample is an RGB image, a per-pixel class mask and the displayed time. Images are
generated from an integer seed, so a dataset of any size needs no storage and the
validation set is reproducible. Variation covers dial and hand styles, numerals, ticks,
perspective tilt, clutter, blur, noise, glare and JPEG artefacts.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

# PIVOT and TOP exist only in the mask: a small disk at the hand pivot and one just inside the
# 12 o'clock mark. They let the geometry recover the projected centre and the dial orientation.
BACKGROUND, DIAL, HOUR, MINUTE, SECOND, PIVOT, TOP = range(7)
CLASS_NAMES = ("background", "dial", "hour", "minute", "second", "pivot", "top")

ROMAN = ("XII", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI")
FONTS = (
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
)


@dataclass
class ClockTime:
    hour: int  # 0..11
    minute: int
    second: int
    has_second_hand: bool

    @property
    def minutes(self) -> float:
        """Position on the 12-hour dial in minutes, 0..720."""
        return self.hour * 60 + self.minute + self.second / 60


@dataclass
class SynthConfig:
    size: int = 256
    max_tilt: float = 0.18  # perspective corner jitter as a fraction of the image size
    max_roll_deg: float = 25.0  # in-plane rotation, recovered through the TOP marker
    second_hand_prob: float = 0.4
    clutter_prob: float = 0.6
    blur_prob: float = 0.3
    jpeg_prob: float = 0.5
    glare_prob: float = 0.25


def hand_direction(angle: float) -> np.ndarray:
    """Unit vector for a clockwise angle from 12 o'clock, in image coordinates (y down)."""
    return np.array([np.sin(angle), -np.cos(angle)])


def time_to_angles(t: ClockTime) -> tuple[float, float, float]:
    hour = (t.hour + t.minute / 60 + t.second / 3600) / 12 * 2 * np.pi
    minute = (t.minute + t.second / 60) / 60 * 2 * np.pi
    second = t.second / 60 * 2 * np.pi
    return hour, minute, second


class ClockRenderer:
    def __init__(self, cfg: SynthConfig | None = None):
        self.cfg = cfg or SynthConfig()

    def render(self, seed: int) -> tuple[np.ndarray, np.ndarray, ClockTime]:
        rng = np.random.default_rng(seed)
        s = self.cfg.size
        t = ClockTime(
            int(rng.integers(12)),
            int(rng.integers(60)),
            int(rng.integers(60)),
            bool(rng.random() < self.cfg.second_hand_prob),
        )

        layer = np.zeros((s, s, 3), np.uint8)
        alpha = np.zeros((s, s), np.uint8)
        mask = np.zeros((s, s), np.uint8)
        radius = rng.uniform(0.26, 0.40) * s
        center = np.array([s / 2, s / 2]) + rng.uniform(-0.08, 0.08, 2) * s
        face = self._draw_dial(rng, layer, alpha, mask, center, radius)
        cv2.circle(mask, _pt(center + hand_direction(0.0) * 0.8 * radius), int(0.08 * radius) + 1, TOP, -1)
        self._draw_hands(rng, layer, mask, center, radius, t, face)
        cv2.circle(mask, _pt(center), int(0.06 * radius) + 1, PIVOT, -1)

        matrix = self._random_homography(rng, center)
        layer = cv2.warpPerspective(layer, matrix, (s, s), flags=cv2.INTER_LINEAR)
        alpha = cv2.warpPerspective(alpha, matrix, (s, s), flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(mask, matrix, (s, s), flags=cv2.INTER_NEAREST)

        background = self._background(rng)
        a = alpha[..., None].astype(np.float32) / 255
        image = (layer * a + background * (1 - a)).astype(np.uint8)
        return self._photometric(rng, image), mask, t

    # --- drawing ----------------------------------------------------------------------------

    def _draw_dial(self, rng, layer, alpha, mask, center, radius) -> np.ndarray:
        c = tuple(np.round(center).astype(int))
        r = int(radius)
        face = rng.integers(170, 256, 3) if rng.random() < 0.75 else rng.integers(0, 256, 3)
        rim_width = int(rng.uniform(0.02, 0.12) * r)
        rim = rng.integers(0, 256, 3)
        cv2.circle(layer, c, r + rim_width, _color(rim), -1, cv2.LINE_AA)
        cv2.circle(alpha, c, r + rim_width, 255, -1, cv2.LINE_AA)
        cv2.circle(mask, c, r + rim_width, DIAL, -1)
        cv2.circle(layer, c, r, _color(face), -1, cv2.LINE_AA)

        ink = _contrasting(rng, face)
        style = rng.choice(["ticks60", "ticks12", "none"], p=[0.5, 0.35, 0.15])
        if style != "none":
            for i in range(60 if style == "ticks60" else 12):
                major = style == "ticks12" or i % 5 == 0
                length = (0.12 if major else 0.05) * r
                d = hand_direction(i / (60 if style == "ticks60" else 12) * 2 * np.pi)
                p0, p1 = center + d * (r * 0.97 - length), center + d * r * 0.97
                cv2.line(
                    layer,
                    _pt(p0),
                    _pt(p1),
                    _color(ink),
                    max(1, int(r * (0.025 if major else 0.01))),
                    cv2.LINE_AA,
                )

        numerals = rng.choice(["arabic", "roman", "quarters", "none"], p=[0.4, 0.2, 0.15, 0.25])
        if numerals != "none":
            font = int(rng.choice(FONTS))
            scale = r / 110 * rng.uniform(0.7, 1.0)
            thickness = max(1, int(round(scale * rng.uniform(1.0, 2.5))))
            ring = r * rng.uniform(0.68, 0.8)
            for h in range(12):
                if numerals == "quarters" and h % 3:
                    continue
                text = ROMAN[h] if numerals == "roman" else str(12 if h == 0 else h)
                (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
                p = center + hand_direction(h / 12 * 2 * np.pi) * ring
                cv2.putText(
                    layer,
                    text,
                    (int(p[0] - tw / 2), int(p[1] + th / 2)),
                    font,
                    scale,
                    _color(ink),
                    thickness,
                    cv2.LINE_AA,
                )
        return face

    def _draw_hands(self, rng, layer, mask, center, radius, t: ClockTime, face: np.ndarray) -> None:
        hour_angle, minute_angle, second_angle = time_to_angles(t)
        ink = _contrasting(rng, face)
        if rng.random() < 0.2:  # occasionally coloured hands, as long as they stay visible
            candidate = rng.integers(0, 256, 3)
            if abs(candidate.mean() - face.mean()) > 70:
                ink = candidate
        style = rng.choice(["bar", "taper", "arrow", "leaf"])
        tail = rng.uniform(0.0, 0.22) * radius
        width = rng.uniform(0.035, 0.075) * radius

        hour_len = rng.uniform(0.45, 0.62) * radius
        minute_len = rng.uniform(max(0.72, hour_len / radius + 0.15), 0.95) * radius
        _fill(layer, mask, _hand_polygon(center, hour_angle, hour_len, tail, width * 1.25, style), ink, HOUR)
        _fill(layer, mask, _hand_polygon(center, minute_angle, minute_len, tail, width, style), ink, MINUTE)
        if t.has_second_hand:
            red = np.array([rng.integers(150, 256), rng.integers(0, 60), rng.integers(0, 60)])
            poly = _hand_polygon(
                center,
                second_angle,
                rng.uniform(0.8, 0.97) * radius,
                tail * 1.4,
                max(1.5, width * 0.3),
                "bar",
            )
            _fill(layer, mask, poly, red if rng.random() < 0.7 else ink, SECOND)

        hub = int(rng.uniform(0.03, 0.08) * radius) + 1
        cv2.circle(layer, _pt(center), hub, _color(ink), -1, cv2.LINE_AA)

    def _random_homography(self, rng, center) -> np.ndarray:
        s = self.cfg.size
        roll = np.deg2rad(rng.uniform(-self.cfg.max_roll_deg, self.cfg.max_roll_deg))
        rot = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), float(np.rad2deg(roll)), 1.0)
        rot = np.vstack([rot, [0, 0, 1]])
        src = np.float32([[0, 0], [s, 0], [s, s], [0, s]])
        dst = src + rng.uniform(-self.cfg.max_tilt, self.cfg.max_tilt, (4, 2)).astype(np.float32) * s
        return cv2.getPerspectiveTransform(src, dst) @ rot

    def _background(self, rng) -> np.ndarray:
        s = self.cfg.size
        c0, c1 = rng.integers(0, 256, 3), rng.integers(0, 256, 3)
        ramp = np.linspace(0, 1, s)[:, None, None]
        if rng.random() < 0.5:
            ramp = ramp.transpose(1, 0, 2)
        bg = c0 * (1 - ramp) + c1 * ramp
        noise = cv2.resize(
            rng.normal(0, 1, (8, 8, 3)).astype(np.float32), (s, s), interpolation=cv2.INTER_CUBIC
        )
        bg = np.clip(bg + noise * rng.uniform(5, 40), 0, 255).astype(np.uint8)
        if rng.random() < self.cfg.clutter_prob:
            for _ in range(rng.integers(3, 12)):
                p0, p1 = _pt(rng.uniform(0, s, 2)), _pt(rng.uniform(0, s, 2))
                color = _color(rng.integers(0, 256, 3))
                kind = rng.integers(3)
                if kind == 0:
                    cv2.rectangle(bg, p0, p1, color, -1)
                elif kind == 1:
                    cv2.line(bg, p0, p1, color, int(rng.integers(1, 6)), cv2.LINE_AA)
                else:
                    cv2.circle(bg, p0, int(rng.uniform(5, s / 6)), color, -1, cv2.LINE_AA)
        return bg

    def _photometric(self, rng, image: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        img = image.astype(np.float32)
        if rng.random() < cfg.glare_prob:
            s = cfg.size
            yy, xx = np.mgrid[0:s, 0:s]
            cx, cy, sigma = rng.uniform(0, s), rng.uniform(0, s), rng.uniform(0.1, 0.35) * s
            glare = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))[..., None]
            img = img + glare * rng.uniform(60, 180)
        img = img * rng.uniform(0.6, 1.3) + rng.uniform(-30, 30)
        img = img * rng.uniform(0.85, 1.15, 3)  # white balance
        img = np.clip(img + rng.normal(0, rng.uniform(0, 8), img.shape), 0, 255).astype(np.uint8)
        if rng.random() < cfg.blur_prob:
            k = int(rng.choice([3, 5, 7]))
            if rng.random() < 0.5:
                img = cv2.GaussianBlur(img, (k, k), 0)
            else:  # motion blur
                kernel = np.zeros((k, k), np.float32)
                kernel[k // 2] = 1 / k
                rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1)
                img = cv2.filter2D(img, -1, cv2.warpAffine(kernel, rot, (k, k)))
        if rng.random() < cfg.jpeg_prob:
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(25, 90))])
            img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        return img


def _hand_polygon(center, angle, length, tail, width, style) -> np.ndarray:
    d = hand_direction(angle)
    n = np.array([-d[1], d[0]])
    if style == "bar":
        pts = [(-tail, -1), (length, -1), (length, 1), (-tail, 1)]
    elif style == "taper":
        pts = [(-tail, -1), (length, -0.15), (length, 0.15), (-tail, 1)]
    elif style == "arrow":
        neck = length * 0.8
        pts = [(-tail, -0.6), (neck, -0.6), (neck, -1.8), (length, 0), (neck, 1.8), (neck, 0.6), (-tail, 0.6)]
    else:  # leaf
        s = np.linspace(-tail, length, 16)
        profile = np.sin(np.pi * (s + tail) / (length + tail)) * 1.4 + 0.2
        pts = [(a, -w) for a, w in zip(s, profile, strict=True)] + [
            (a, w) for a, w in zip(s[::-1], profile[::-1], strict=True)
        ]
    return np.array([center + a * d + w * width / 2 * n for a, w in pts])


def _fill(layer, mask, polygon, color, cls) -> None:
    pts = np.round(polygon).astype(np.int32)
    cv2.fillPoly(layer, [pts], _color(color), cv2.LINE_AA)
    cv2.fillPoly(mask, [pts], int(cls))


def _contrasting(rng, color) -> np.ndarray:
    return rng.integers(0, 60, 3) if np.mean(color) > 110 else rng.integers(200, 256, 3)


def _color(c) -> tuple[int, int, int]:
    return tuple(int(v) for v in c)


def _pt(p) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


def main(argv: list[str] | None = None) -> None:
    """Write a folder of rendered samples for inspection: image, mask preview and labels."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/synth_preview"))
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    renderer = ClockRenderer()
    palette = np.array(
        [[0, 0, 0], [90, 90, 90], [0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 255, 255], [255, 0, 255]],
        np.uint8,
    )
    labels = {}
    for i in range(args.n):
        image, mask, t = renderer.render(args.seed + i)
        preview = np.hstack([cv2.cvtColor(image, cv2.COLOR_RGB2BGR), palette[mask]])
        cv2.imwrite(str(args.out / f"{i:04d}.png"), preview)
        labels[f"{i:04d}.png"] = asdict(t)
    (args.out / "labels.json").write_text(json.dumps(labels, indent=1))
    print(f"wrote {args.n} samples to {args.out}")


if __name__ == "__main__":
    main()
