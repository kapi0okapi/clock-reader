"""Figures for the README and the portfolio site.

    python scripts/make_figures.py --checkpoint runs/resnet34_unet/best.pt --out assets

Writes
- ``reading.gif``: hard test clocks (tilted, rotated, cluttered) read step by step: the photo,
  the U-Net masks with the hands as the geometry reads them, and the dial rectified back to a
  frontal view with 12 o'clock up;
- ``results.png``: share of clocks read within 5 minutes per method, and the augmentation
  ablation, from the ``eval.json`` files next to each checkpoint.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from clock_reader.classical import classical_mask
from clock_reader.evaluate import TEST_SEED
from clock_reader.geometry import Reading, circular_minutes_error, read_time
from clock_reader.inference import ClockReader
from clock_reader.synth import HOUR, MINUTE, PIVOT, SECOND, TOP, ClockRenderer

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = (14, 16, 19)
MUTED = (106, 112, 120)
LINE = (227, 230, 233)
ARROW = (201, 205, 210)
ACCENT = (255, 79, 26)
CYAN = (10, 158, 194)
PURPLE = (150, 70, 210)
GOOD = (34, 139, 76)
MASK_COLORS = {HOUR: ACCENT, MINUTE: CYAN, SECOND: (106, 163, 24), TOP: PURPLE, PIVOT: INK}
PANEL, GAP, MARGIN, HEAD, FOOT = 320, 64, 36, 74, 84


def font(path: str | None, size: int, weight: int = 400) -> ImageFont.FreeTypeFont:
    if path is None:
        path = matplotlib.font_manager.findfont("DejaVu Sans")
    f = ImageFont.truetype(path, size)
    with contextlib.suppress(OSError):  # weight only applies to variable fonts such as Geist
        f.set_variation_by_axes([weight])
    return f


def clock_text(minutes: float) -> str:
    hour, minute = divmod(int(round(minutes)) % 720, 60)
    return f"{hour or 12}:{minute:02d}"


def overlay(rgb: np.ndarray, mask: np.ndarray, reading: Reading) -> np.ndarray:
    """The photo, faded, with the predicted hands, pivot and 12 o'clock marker in solid colour."""
    scale = PANEL / rgb.shape[0]
    base = cv2.resize(rgb, (PANEL, PANEL), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    out = 0.4 * base + 0.6 * 255
    big = cv2.resize(mask, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
    for cls, color in MASK_COLORS.items():
        alpha = cv2.GaussianBlur((big == cls).astype(np.float32), (0, 0), 0.9)[..., None]  # soft edges
        out = out * (1 - alpha) + np.array(color, np.float32) * alpha
    out = out.clip(0, 255).astype(np.uint8)
    (cx, cy), axes, angle = reading.ellipse
    ellipse = ((cx * scale, cy * scale), (axes[0] * scale, axes[1] * scale), angle)
    cv2.ellipse(out, ellipse, MUTED, 2, cv2.LINE_AA)
    return out


def rectified(rgb: np.ndarray, reading: Reading) -> np.ndarray:
    """The dial warped to a frontal view, 12 o'clock up, with the hands as the geometry reads them."""
    size, half = PANEL, PANEL / 2
    radius = half / 1.1  # the fitted dial outline, i.e. the unit circle of the rectified frame
    r = np.deg2rad(reading.reference_deg)
    rotate = np.array([[np.cos(r), np.sin(r), 0], [-np.sin(r), np.cos(r), 0], [0, 0, 1]])
    to_pixels = np.array([[radius, 0, half], [0, radius, half], [0, 0, 1]])
    warped = cv2.warpPerspective(rgb, to_pixels @ rotate @ reading.homography, (size, size)).astype(
        np.float32
    )
    yy, xx = np.mgrid[:size, :size]
    alpha = np.clip(radius - np.hypot(xx - half + 0.5, yy - half + 0.5) + 0.5, 0, 1)[..., None]
    out = (warped * alpha + 255 * (1 - alpha)).astype(np.uint8)  # anti-aliased circular crop
    cv2.circle(out, (int(half), int(half)), int(radius), LINE, 2, cv2.LINE_AA)
    centre = np.array([half, half])
    for name, length, width, color in (("hour", 0.5, 7, ACCENT), ("minute", 0.78, 5, CYAN)):
        a = np.deg2rad(reading.angles_deg[name])
        tip = centre + length * radius * np.array([np.sin(a), -np.cos(a)])
        cv2.line(out, tuple(centre.astype(int)), tuple(tip.astype(int)), color, width, cv2.LINE_AA)
        cv2.circle(out, tuple(tip.astype(int)), width // 2, color, -1, cv2.LINE_AA)  # round tip
    cv2.circle(out, (int(half), int(half)), 7, INK, -1, cv2.LINE_AA)
    cv2.circle(out, (int(half), int(half)), 2, (255, 255, 255), -1, cv2.LINE_AA)
    return out


def pick_examples(reader: ClockReader, renderer: ClockRenderer, count: int) -> list[tuple]:
    """Hard clocks the network reads within 2 minutes. "Hard" = the classical Hough pipeline cannot read
    them, a cheap proxy for strong tilt, roll or clutter; its result is not shown."""
    examples, seed = [], TEST_SEED
    while len(examples) < count and seed < TEST_SEED + 2000:
        image, _, truth = renderer.render(seed)
        seed += 1
        reading, mask = reader.read(image)
        if reading is None or circular_minutes_error(reading.minutes_float, truth.minutes) > 2:
            continue
        classical = read_time(classical_mask(image))
        if classical is not None and circular_minutes_error(classical.minutes_float, truth.minutes) <= 15:
            continue
        examples.append((image, mask, reading, truth))
    return examples


def tick(draw: ImageDraw.ImageDraw, x: float, y: float, color: tuple) -> None:
    """A drawn check mark (not every font has the glyph), vertically centred on y."""
    draw.line([(x, y), (x + 7, y + 8), (x + 21, y - 8)], fill=color, width=4, joint="curve")


def rounded(panel: np.ndarray, radius: int = 14) -> tuple[Image.Image, Image.Image]:
    shape = Image.new("L", (PANEL, PANEL), 0)
    ImageDraw.Draw(shape).rounded_rectangle([0, 0, PANEL - 1, PANEL - 1], radius, fill=255)
    return Image.fromarray(panel), shape


def frame(example: tuple, fonts: dict) -> Image.Image:
    image, mask, reading, truth = example
    width = 2 * MARGIN + 3 * PANEL + 2 * GAP
    canvas = Image.new("RGB", (width, HEAD + PANEL + FOOT), "white")
    draw = ImageDraw.Draw(canvas)
    panels = [
        ("Photo", cv2.resize(image, (PANEL, PANEL), interpolation=cv2.INTER_CUBIC)),
        ("Segmentation", overlay(image, mask, reading)),
        ("Rectified dial", rectified(image, reading)),
    ]
    xs = [MARGIN + i * (PANEL + GAP) for i in range(3)]
    for i, ((title, panel), x) in enumerate(zip(panels, xs, strict=True)):
        picture, shape = rounded(panel)
        canvas.paste(picture, (x, HEAD), shape)
        draw.rounded_rectangle([x, HEAD, x + PANEL - 1, HEAD + PANEL - 1], 14, outline=LINE, width=2)
        cy = HEAD - 30
        draw.ellipse([x, cy - 13, x + 26, cy + 13], fill=ACCENT)
        draw.text((x + 13, cy), str(i + 1), font=fonts["step"], fill="white", anchor="mm")
        draw.text((x + 38, cy), title, font=fonts["title"], fill=INK, anchor="lm")
        if i < 2:  # arrow to the next step
            ax, ay = x + PANEL + GAP / 2, HEAD + PANEL / 2
            draw.line([(ax - 14, ay), (ax + 12, ay)], fill=ARROW, width=4)
            draw.line(
                [(ax + 2, ay - 10), (ax + 13, ay), (ax + 2, ay + 10)], fill=ARROW, width=4, joint="curve"
            )

    base = HEAD + PANEL + 46  # common baseline of the captions
    label, value = fonts["label"], fonts["value"]
    draw.text((xs[0], base), "true time", font=label, fill=MUTED, anchor="ls")
    draw.text(
        (xs[0] + draw.textlength("true time", font=label) + 12, base),
        clock_text(truth.minutes),
        font=value,
        fill=INK,
        anchor="ls",
    )
    x = xs[1]
    legend = fonts["legend"]
    for text, color in (
        ("hour", ACCENT),
        ("minute", CYAN),
        ("second", MASK_COLORS[SECOND]),
        ("12 o'clock", PURPLE),
    ):
        draw.ellipse([x, base - 13, x + 11, base - 2], fill=color)
        draw.text((x + 16, base), text, font=legend, fill=MUTED, anchor="ls")
        x += 16 + draw.textlength(text, font=legend) + 13
    read = clock_text(reading.minutes_float)
    draw.text((xs[2], base), "read", font=label, fill=MUTED, anchor="ls")
    x = xs[2] + draw.textlength("read", font=label) + 12
    draw.text((x, base), read, font=value, fill=GOOD, anchor="ls")
    tick(draw, x + draw.textlength(read, font=value) + 14, base - 11, GOOD)
    return canvas


def gif_frame(image: Image.Image) -> Image.Image:
    """256-colour frame whose palette always holds the figure's own colours exactly (an adaptive
    palette alone would merge the small 12 o'clock marker into grey on some frames)."""
    fixed = [INK, MUTED, LINE, ARROW, ACCENT, CYAN, PURPLE, GOOD, MASK_COLORS[SECOND], (255, 255, 255)]
    adaptive = image.quantize(colors=256 - len(fixed), method=Image.Quantize.MEDIANCUT)
    palette = adaptive.getpalette()[: 3 * (256 - len(fixed))] + [v for color in fixed for v in color]
    holder = Image.new("P", (1, 1))
    holder.putpalette(palette)
    return image.quantize(palette=holder, dither=Image.Dither.FLOYDSTEINBERG)


def write_mp4(frames: list[Image.Image], path: Path, seconds: float) -> None:
    """The same sequence as a small H.264 video for the web, from the full-colour frames."""
    with tempfile.TemporaryDirectory() as tmp:
        for i, f in enumerate(frames):
            f.save(Path(tmp) / f"{i:03d}.png")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                f"{1 / seconds}",
                "-i",
                f"{tmp}/%03d.png",
                "-vf",
                "scale=trunc(iw/2)*2:trunc(ih/2)*2,fps=25",
                "-c:v",
                "libx264",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            check=True,
        )


def results_chart(runs: Path, out: Path, font_path: str | None) -> None:
    def acc(run: str, split: str, method: str = "network") -> float:
        return 100 * json.loads((runs / run / "eval.json").read_text())[split][method]["acc_5min"]

    rows_method = [
        ("Classical (Hough + Otsu)", acc("resnet34_unet", "synthetic", "classical"), False),
        ("U-Net, ResNet-18", acc("resnet18_unet", "synthetic"), False),
        ("U-Net, ResNet-34", acc("resnet34_unet", "synthetic"), True),
        ("Ground-truth masks (upper bound)", acc("resnet34_unet", "synthetic", "oracle"), False),
    ]
    rows_ablation = [
        ("Full augmentation", acc("resnet34_unet", "synthetic"), True),
        ("Without perspective tilt", acc("ablation_no_tilt", "synthetic"), False),
        ("Without in-plane roll", acc("ablation_no_roll", "synthetic"), False),
    ]
    family = "DejaVu Sans"
    if font_path:
        matplotlib.font_manager.fontManager.addfont(font_path)
        family = matplotlib.font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 11,
            "text.color": "#0e1013",
            "axes.edgecolor": "#c9cdd2",
            "axes.labelcolor": "#6a7078",
            "xtick.color": "#6a7078",
            "ytick.color": "#0e1013",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.1), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.25, wspace=0.12)
    for ax, rows, title in (
        (axes[0], rows_method, "Methods"),
        (axes[1], rows_ablation, "What the augmentation buys"),
    ):
        names, values, highlight = zip(*rows, strict=True)
        colors = ["#ff4f1a" if h else "#d5d9dd" for h in highlight]
        bars = ax.barh(range(len(rows)), values, color=colors, height=0.62)
        ax.set_yticks(range(len(rows)), names)
        ax.tick_params(axis="y", length=0, pad=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100 %"])
        ax.grid(axis="x", color="#eceef0", linewidth=1)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=13.5, pad=12)
        for bar, v, h in zip(bars, values, highlight, strict=True):
            inside = v > 20
            ax.text(
                v - 1.5 if inside else v + 1.5,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.1f} %",
                va="center",
                ha="right" if inside else "left",
                color="white" if h else "#0e1013",
            )
    fig.supxlabel(
        "Share of clocks read within 5 minutes · 2000 unseen synthetic clocks with perspective tilt,"
        " ±25° roll, clutter, blur and glare",
        x=0.01,
        ha="left",
        fontsize=9.5,
        color="#6a7078",
    )
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/resnet34_unet/best.pt"))
    parser.add_argument("--out", type=Path, default=Path("assets"))
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--font", help="TTF for the labels (default: DejaVu Sans)")
    parser.add_argument("--mp4", type=Path, help="also write the sequence as an H.264 video (needs ffmpeg)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    reader = ClockReader(args.checkpoint)
    examples = pick_examples(reader, ClockRenderer(reader.cfg.synth), args.frames)
    fonts = {
        "step": font(args.font, 15, 700),
        "title": font(args.font, 21, 600),
        "label": font(args.font, 18),
        "legend": font(args.font, 16),
        "value": font(args.font, 28, 600),
    }
    frames = [frame(e, fonts) for e in examples]
    palette_frames = [gif_frame(f) for f in frames]
    palette_frames[0].save(
        args.out / "reading.gif", save_all=True, append_images=palette_frames[1:], duration=1600, loop=0
    )
    frames[0].save(args.out / "reading.png")
    if args.mp4 and shutil.which("ffmpeg"):
        write_mp4(frames, args.mp4, seconds=1.6)
    results_chart(args.checkpoint.parent.parent, args.out / "results.png", args.font)
    print(f"wrote {len(frames)} frames to {args.out / 'reading.gif'} and {args.out / 'results.png'}")


if __name__ == "__main__":
    main()
