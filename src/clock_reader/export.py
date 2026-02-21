"""Export the segmentation network to ONNX and check it against PyTorch.

    python -m clock_reader.export --checkpoint runs/resnet34_unet/best.pt

The geometric reading stays in Python/NumPy; ONNX covers the network only (input: normalised
1x3xSxS float32, output: per-class logits).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from clock_reader.inference import ClockReader


def export_onnx(checkpoint: Path, output: Path, opset: int = 18) -> Path:
    reader = ClockReader(checkpoint)
    size = reader.size
    torch.onnx.export(
        reader.model,
        (torch.randn(2, 3, size, size),),
        str(output),
        input_names=["image"],
        output_names=["logits"],
        dynamic_shapes=({0: torch.export.Dim("batch")},),
        opset_version=opset,
    )
    output.with_suffix(".json").write_text(
        json.dumps({"image_size": size, "mean": reader.mean, "std": reader.std}, indent=2)
    )

    import onnxruntime as ort

    x = np.random.default_rng(0).standard_normal((3, 3, size, size)).astype(np.float32)
    onnx_out = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"]).run(None, {"image": x})[
        0
    ]
    with torch.no_grad():
        torch_out = reader.model(torch.from_numpy(x)).numpy()
    agreement = float((onnx_out.argmax(1) == torch_out.argmax(1)).mean())
    if agreement < 0.999:
        raise RuntimeError(f"ONNX and PyTorch masks agree on only {agreement:.4%} of pixels")
    print(f"wrote {output} (pixel agreement with PyTorch {agreement:.4%})")
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    export_onnx(args.checkpoint, args.output or args.checkpoint.with_name("clock_segnet.onnx"))


if __name__ == "__main__":
    main()
