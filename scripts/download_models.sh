#!/bin/bash
# Pre-fetch pretrained weights on a login node (compute nodes may not have internet access):
# the timm encoders, and the YOLO COCO detector used by `clock-read --detect`.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models
python - <<'PY'
import timm
from ultralytics import YOLO

for name in ("resnet34", "resnet18"):
    timm.create_model(name, pretrained=True, features_only=True)
    print("cached", name)
YOLO("models/yolo26s.pt")
print("cached models/yolo26s.pt")
PY
