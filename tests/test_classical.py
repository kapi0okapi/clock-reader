import numpy as np

from clock_reader.classical import classical_mask
from clock_reader.geometry import circular_minutes_error, read_time
from clock_reader.synth import ClockRenderer, SynthConfig


def test_classical_reads_easy_clocks():
    easy = SynthConfig(max_tilt=0, max_roll_deg=0, clutter_prob=0, blur_prob=0, glare_prob=0, jpeg_prob=0)
    renderer = ClockRenderer(easy)
    hits = 0
    for i in range(40):
        image, _, t = renderer.render(100 + i)
        reading = read_time(classical_mask(image))
        hits += reading is not None and circular_minutes_error(reading.minutes_float, t.minutes) <= 5
    assert hits / 40 > 0.5


def test_classical_mask_without_clock_is_empty():
    assert classical_mask(np.full((128, 128, 3), 127, np.uint8)).max() == 0
