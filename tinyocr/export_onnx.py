"""Export the checkpoint to a fixed-shape ONNX graph and a JSON spec."""

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import onnx
import torch

from tinyocr.charset import CHARSET
from tinyocr.config import BLANK, DETAIL_H, DETAIL_W, IMG_H, IMG_W
from tinyocr.model import recognizer_for


def export_checkpoint(checkpoint: Path, out: Path) -> None:
    try:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location="cpu")
    height = ckpt.get("height")
    width = ckpt.get("width")
    if ckpt.get("charset") != CHARSET or (height, width) not in (
        (IMG_H, IMG_W),
        (DETAIL_H, DETAIL_W),
    ):
        raise SystemExit("checkpoint contract does not match this package config")
    model = recognizer_for(int(height), len(CHARSET) + 1)
    model.load_state_dict(ckpt["model"])
    model.eval()
    dummy = torch.zeros(1, 1, int(height), int(width), dtype=torch.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        input_names=["image"],
        output_names=["log_probs"],
        opset_version=17,
        do_constant_folding=True,
        external_data=False,
        dynamo=False,
        dynamic_axes={"image": {0: "batch"}, "log_probs": {0: "batch"}},
    )
    export_params = inspect.signature(torch.onnx.export).parameters
    kwargs = {key: value for key, value in kwargs.items() if key in export_params}
    with torch.no_grad():
        reference = model(dummy).numpy()
        torch.onnx.export(model, (dummy,), str(out), **kwargs)
    onnx_model = onnx.load(str(out))
    onnx.checker.check_model(onnx_model)
    meta = {
        "charset": CHARSET,
        "blank": BLANK,
        "height": int(height),
        "width": int(width),
        "input_name": "image",
        "output_name": "log_probs",
        "val_exact": ckpt.get("val_exact"),
        "val_character_accuracy": ckpt.get("val_character_accuracy"),
    }
    for key, value in meta.items():
        entry = onnx_model.metadata_props.add()
        entry.key = key
        entry.value = json.dumps(value)
    onnx.save(onnx_model, str(out))
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    import onnxruntime as ort

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    exported = session.run(["log_probs"], {"image": dummy.numpy()})[0]
    if exported.shape != reference.shape:
        raise SystemExit(f"onnx shape {exported.shape} != torch shape {reference.shape}")
    if not np.allclose(exported, reference, rtol=1e-4, atol=1e-4):
        gap = float(np.max(np.abs(exported - reference)))
        raise SystemExit(f"onnx output diverges from pytorch, max abs {gap}")
    print(f"exported {out}  output {tuple(exported.shape)}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the recognizer to ONNX")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/recognizer.pt"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/recognizer.onnx"))
    args = parser.parse_args(argv)
    export_checkpoint(args.checkpoint, args.out)
    return 0
