"""End-to-end smoke test: one tiny epoch, evaluation, ONNX export, prediction."""

import json
from pathlib import Path

import cv2

from clock_reader import evaluate, export, predict
from clock_reader.config import TrainConfig
from clock_reader.synth import ClockRenderer
from clock_reader.train import train
from clock_reader.utils.config import load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_train_evaluate_export_predict(tmp_path):
    cfg = load_config(TrainConfig, CONFIGS / "smoke.yaml")
    run_dir = tmp_path / "run"
    best = train(cfg, run_dir)
    assert "acc_5min" in best and (run_dir / "best.pt").exists()

    evaluate.main(["--checkpoint", str(run_dir / "best.pt"), "--n", "8"])
    report = json.loads((run_dir / "eval.json").read_text())
    assert set(report["synthetic"]) == {"classical", "network", "oracle"}
    assert report["synthetic"]["oracle"]["acc_5min"] == 1.0

    assert export.export_onnx(run_dir / "best.pt", tmp_path / "net.onnx").exists()

    image, _, _ = ClockRenderer().render(1)
    cv2.imwrite(str(tmp_path / "clock.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    predict.main(
        [
            "--checkpoint",
            str(run_dir / "best.pt"),
            "--source",
            str(tmp_path / "clock.png"),
            "--out",
            str(tmp_path / "pred"),
        ]
    )
    assert (tmp_path / "pred" / "clock.png").exists()
