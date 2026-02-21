"""Time from a segmentation mask, without any learned time model.

1. Fit an ellipse (a conic) to the dial outline.
2. Locate the projected dial centre: the PIVOT class, else the intersection of the hand axes,
   else the ellipse centre.
3. Rectify. The centre of a circle is the pole of the line at infinity, so the polar line of the
   projected centre with respect to the ellipse is the vanishing line of the dial plane. Sending
   it back to infinity removes the perspective; a final affine map turns the ellipse into the
   unit circle. The dial is then metrically correct up to a rotation.
4. Fix that rotation with the TOP class (12 o'clock); without it, assume the clock is upright.
5. Read the minutes from the minute hand, then pick the hour whose expected hour-hand angle
   (hour + minutes/60) best matches the hour hand. This fixes the classic "2:58 read as 3:58".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from clock_reader.synth import HOUR, MINUTE, PIVOT, SECOND, TOP

TWO_PI = 2 * np.pi


@dataclass
class Reading:
    hour: int  # 0..11 (0 is shown as 12)
    minute: int
    second: int | None
    minutes_float: float  # continuous position on the 12 h dial, 0..720
    confidence: float
    pivot: tuple[float, float]  # image coordinates
    ellipse: tuple
    angles_deg: dict[str, float] = field(default_factory=dict)  # clockwise from the dial's 12 o'clock
    consistency_deg: float = 0.0  # hour hand vs. the angle implied by the minute hand
    homography: np.ndarray | None = None  # image -> rectified unit-circle frame
    reference_deg: float = 0.0  # direction of 12 o'clock in the rectified frame

    def __str__(self) -> str:
        text = f"{self.hour or 12}:{self.minute:02d}"
        return text + (f":{self.second:02d}" if self.second is not None else "")


def hand_tip(reading: Reading, hand: str, length: float) -> np.ndarray:
    """Image coordinates of a point ``length`` dial radii along a hand, for drawing."""
    angle = np.deg2rad(reading.angles_deg[hand] + reading.reference_deg)
    point = np.array([[np.sin(angle), -np.cos(angle)]]) * length
    return apply_homography(np.linalg.inv(reading.homography), point)[0]


def circular_minutes_error(a: float, b: float, period: float = 720.0) -> float:
    d = abs(a - b) % period
    return min(d, period - d)


# --- dial and rectification ----------------------------------------------------------------


def fit_dial(mask: np.ndarray) -> tuple | None:
    """Ellipse ((cx, cy), (width, height), angle) fitted to the outline of the largest clock blob.
    Outline points on the image border are ignored, so partly cropped clocks still fit."""
    fg = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n < 2:
        return None
    blob = (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    points = max(contours, key=cv2.contourArea)[:, 0, :]
    h, w = mask.shape
    inner = (points[:, 0] > 1) & (points[:, 0] < w - 2) & (points[:, 1] > 1) & (points[:, 1] < h - 2)
    points = points[inner] if inner.sum() >= 0.5 * len(points) else points
    if len(points) < 5:
        return None
    return cv2.fitEllipse(points.astype(np.float32))


def ellipse_to_conic(ellipse: tuple) -> np.ndarray:
    """3x3 symmetric matrix C with x^T C x = 0 for homogeneous points x on the ellipse."""
    (cx, cy), (w, h), angle = ellipse
    t = np.deg2rad(angle)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    q = rot @ np.diag([4 / w**2, 4 / h**2]) @ rot.T
    c = np.array([cx, cy])
    conic = np.zeros((3, 3))
    conic[:2, :2] = q
    conic[:2, 2] = conic[2, :2] = -q @ c
    conic[2, 2] = c @ q @ c - 1
    return conic


def rectifying_homography(ellipse: tuple, center: np.ndarray) -> np.ndarray:
    """Homography mapping the dial to the unit circle with the projected ``center`` at the origin."""
    translate = np.array([[1, 0, -center[0]], [0, 1, -center[1]], [0, 0, 1.0]])
    t_inv = np.linalg.inv(translate)
    conic = t_inv.T @ ellipse_to_conic(ellipse) @ t_inv
    polar = conic[:, 2]  # polar line of the origin = vanishing line of the dial plane
    projective = np.eye(3)
    projective[2, :2] = polar[:2] / polar[2]
    p_inv = np.linalg.inv(projective)
    conic = p_inv.T @ conic @ p_inv  # now an ellipse centred at the origin
    shape = conic[:2, :2] / -conic[2, 2]
    eigval, eigvec = np.linalg.eigh(shape)
    if np.any(eigval <= 0):  # degenerate fit; keep only the translation
        return translate
    if np.linalg.det(eigvec) < 0:  # a proper rotation, otherwise the dial is mirrored
        eigvec[:, 1] *= -1
    affine = np.eye(3)
    affine[:2, :2] = np.diag(np.sqrt(eigval)) @ eigvec.T
    return affine @ projective @ translate


def apply_homography(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.c_[points, np.ones(len(points))] @ h.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def angle_from_up(v: np.ndarray) -> np.ndarray:
    """Clockwise angle from the negative y axis (12 o'clock in image convention)."""
    return np.arctan2(v[..., 0], -v[..., 1]) % TWO_PI


# --- hands -----------------------------------------------------------------------------------


def hand_axes_intersection(hands: list[np.ndarray]) -> np.ndarray | None:
    normals, offsets, directions = [], [], []
    for pts in hands:
        if len(pts) < 10:
            continue
        vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        normals.append([-vy, vx])
        offsets.append(-vy * x0 + vx * y0)
        directions.append((vx, vy))
    if len(normals) < 2:
        return None
    (ax, ay), (bx, by) = directions[:2]
    if abs(ax * by - ay * bx) < np.sin(np.deg2rad(20)):  # nearly parallel: unstable
        return None
    point, *_ = np.linalg.lstsq(np.array(normals), np.array(offsets), rcond=None)
    return point


def hand_angle(points: np.ndarray) -> tuple[float, float]:
    """Angle (clockwise from up) and length of a hand in the rectified frame, pivot at the origin.
    Pixels are weighted by squared distance, so the tip outweighs a short counterweight tail."""
    r = np.linalg.norm(points, axis=1)
    keep = r > 0.08 * r.max()
    if keep.sum() < 3:
        keep = r > 0
    direction = (points[keep] * r[keep, None]).sum(0)  # unit vectors weighted by r^2
    return float(angle_from_up(direction)), float(np.percentile(r, 98))


def split_hands(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Separates unlabelled hand pixels (rectified, pivot at origin) into (hour, minute): the two
    strongest directions are the hands, and the longer one is the minute hand."""
    r = np.linalg.norm(points, axis=1)
    theta = angle_from_up(points)
    bins = 72
    hist = np.bincount((theta / TWO_PI * bins).astype(int) % bins, weights=r, minlength=bins)
    hist = np.convolve(np.r_[hist[-2:], hist, hist[:2]], np.ones(5) / 5, mode="valid")
    first = int(np.argmax(hist))
    masked = hist.copy()
    masked[[(first + k) % bins for k in range(-6, 7)]] = 0  # suppress +-30 degrees around it
    second = int(np.argmax(masked))
    if masked[second] < 0.15 * hist[first]:  # hands overlap: a single direction
        return points, points
    centers = (np.array([first, second]) + 0.5) / bins * TWO_PI
    dist = np.abs((theta[:, None] - centers[None, :] + np.pi) % TWO_PI - np.pi)
    assign = np.argmin(dist, axis=1)
    groups = [points[assign == 0], points[assign == 1]]
    lengths = [np.percentile(np.linalg.norm(g, axis=1), 98) if len(g) else 0 for g in groups]
    minute_idx = int(np.argmax(lengths))
    return groups[1 - minute_idx], groups[minute_idx]


def angles_to_time(hour_angle: float, minute_angle: float) -> tuple[int, float, float]:
    """(hour 0..11, minutes 0..60 as float, consistency error in degrees)."""
    minutes = minute_angle / TWO_PI * 60
    expected = (np.arange(12) + minutes / 60) / 12 * TWO_PI
    diff = np.abs((expected - hour_angle + np.pi) % TWO_PI - np.pi)
    hour = int(np.argmin(diff))
    return hour, float(minutes), float(np.rad2deg(diff[hour]))


# --- reading -----------------------------------------------------------------------------------


def _pixels(mask: np.ndarray, cls: int) -> np.ndarray:
    ys, xs = np.nonzero(mask == cls)
    return np.stack([xs, ys], axis=1).astype(np.float64)


def read_time(mask: np.ndarray) -> Reading | None:
    ellipse = fit_dial(mask)
    if ellipse is None:
        return None
    radius = max(ellipse[1]) / 2
    min_pixels = max(8, int(1e-3 * mask.size))
    hour_px, minute_px, second_px = (_pixels(mask, c) for c in (HOUR, MINUTE, SECOND))
    pivot_px, top_px = _pixels(mask, PIVOT), _pixels(mask, TOP)

    center = np.array(ellipse[0])
    if len(pivot_px) >= 4:
        center = pivot_px.mean(0)
    elif (crossing := hand_axes_intersection([hour_px, minute_px])) is not None and np.linalg.norm(
        crossing - center
    ) < 0.25 * radius:
        center = crossing
    rectify = rectifying_homography(ellipse, center)

    if len(top_px) >= 4:  # 12 o'clock marker -> reference direction
        reference = float(angle_from_up(apply_homography(rectify, top_px).mean(0)))
    else:  # assume an upright clock: image "up" through the centre
        up = apply_homography(rectify, np.array([center - [0, 0.5 * radius]]))[0]
        reference = float(angle_from_up(up))

    merged = len(hour_px) < min_pixels or len(minute_px) < min_pixels
    if merged:
        hands = np.concatenate([hour_px, minute_px])
        if len(hands) < min_pixels:
            return None
        hour_r, minute_r = split_hands(apply_homography(rectify, hands))
    else:
        hour_r, minute_r = apply_homography(rectify, hour_px), apply_homography(rectify, minute_px)

    hour_angle = (hand_angle(hour_r)[0] - reference) % TWO_PI
    minute_angle = (hand_angle(minute_r)[0] - reference) % TWO_PI
    hour, minutes, consistency = angles_to_time(hour_angle, minute_angle)
    angles = {"hour": np.rad2deg(hour_angle), "minute": np.rad2deg(minute_angle)}

    second = None
    if len(second_px) >= min_pixels // 2:
        second_angle = (hand_angle(apply_homography(rectify, second_px))[0] - reference) % TWO_PI
        second = int(second_angle / TWO_PI * 60) % 60
        angles["second"] = np.rad2deg(second_angle)

    total = hour * 60 + minutes
    confidence = float(np.exp(-consistency / 15.0)) * (0.7 if merged else 1.0)
    return Reading(
        hour=int(total // 60) % 12,
        minute=int(total % 60),
        second=second,
        minutes_float=float(total % 720),
        confidence=round(confidence, 4),
        pivot=(float(center[0]), float(center[1])),
        ellipse=ellipse,
        angles_deg={k: round(float(v), 2) for k, v in angles.items()},
        consistency_deg=round(consistency, 2),
        homography=rectify,
        reference_deg=float(np.rad2deg(reference)),
    )
