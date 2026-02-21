import cv2
import numpy as np
import pytest

from clock_reader.geometry import (
    angles_to_time,
    apply_homography,
    circular_minutes_error,
    read_time,
    rectifying_homography,
)
from clock_reader.synth import (
    CLASS_NAMES,
    HOUR,
    MINUTE,
    ClockRenderer,
    ClockTime,
    SynthConfig,
    time_to_angles,
)


def test_rendering_is_deterministic():
    renderer = ClockRenderer()
    a, mask_a, t_a = renderer.render(42)
    b, mask_b, t_b = renderer.render(42)
    assert np.array_equal(a, b) and np.array_equal(mask_a, mask_b) and t_a == t_b
    assert mask_a.max() < len(CLASS_NAMES)
    assert (mask_a == HOUR).any() and (mask_a == MINUTE).any()


def test_time_to_angles():
    hour, minute, second = time_to_angles(ClockTime(3, 30, 0, False))
    assert np.isclose(np.rad2deg(hour), 105) and np.isclose(np.rad2deg(minute), 180) and second == 0


def test_hour_is_corrected_by_minutes():
    # 2:58 - the hour hand points almost exactly at 3, but the minute hand says "just before"
    hour_angle = np.deg2rad((2 + 58 / 60) * 30 + 1.0)  # slightly past 3 o'clock due to noise
    hour, minutes, consistency = angles_to_time(hour_angle, np.deg2rad(58 * 6))
    assert hour == 2 and round(minutes) == 58 and consistency < 2


@pytest.mark.parametrize("angle", [0, 30, 75, 120])
def test_rectification_maps_ellipse_to_unit_circle(angle):
    ellipse = ((120.0, 90.0), (160.0, 90.0), float(angle))
    points = cv2.ellipse2Poly((120, 90), (80, 45), angle, 0, 360, 10).astype(float)
    mapped = apply_homography(rectifying_homography(ellipse, np.array([120.0, 90.0])), points)
    np.testing.assert_allclose(np.linalg.norm(mapped, axis=1), 1.0, atol=0.03)


def test_geometry_on_ground_truth_masks():
    """With perfect masks, the geometric step alone must read almost every clock to the minute,
    including tilt and +-25 degree roll."""
    renderer = ClockRenderer(SynthConfig())
    errors = []
    for i in range(150):
        _, mask, t = renderer.render(7_000 + i)
        reading = read_time(mask)
        errors.append(circular_minutes_error(reading.minutes_float, t.minutes))
    assert np.mean(np.array(errors) <= 1) > 0.97


def test_circular_error_wraps():
    assert circular_minutes_error(5, 715) == 10
