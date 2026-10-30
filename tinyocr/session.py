"""ONNX Runtime session. Inference does not import PyTorch."""

import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from tinyocr.decode import greedy_decode
from tinyocr.preprocess import prepare_line


def load_spec(model_path: Path) -> dict:
    sidecar = model_path.with_suffix(".json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"missing model spec: {sidecar}")
    spec = json.loads(sidecar.read_text(encoding="utf-8"))
    for key in ("charset", "height", "width", "input_name", "output_name", "blank"):
        if key not in spec:
            raise ValueError(f"{sidecar} is missing {key}")
    return spec


def resolve_threads(threads: int) -> int:
    """0 means a small pool: more than 4 threads does not help this model."""
    if threads == 0:
        return min(4, os.cpu_count() or 1)
    if threads < 1:
        raise ValueError("threads must be >= 1, or 0 to choose automatically")
    return threads


def load_session(model_path: Path, threads: int = 0) -> ort.InferenceSession:
    threads = resolve_threads(threads)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def predict_batch(session: ort.InferenceSession, spec: dict, batch: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(batch, dtype=np.float32)
    expected = (array.shape[0], 1, int(spec["height"]), int(spec["width"]))
    if array.shape != expected:
        raise ValueError(f"expected images shaped {expected}, got {array.shape}")
    return session.run([spec["output_name"]], {spec["input_name"]: array})[0]


def predict_image(
    session: ort.InferenceSession,
    spec: dict,
    image: Image.Image,
) -> str:
    tensor = prepare_line(image, int(spec["height"]), int(spec["width"]))
    log_probs = predict_batch(session, spec, tensor[None, ...])
    return greedy_decode(log_probs, spec["charset"])[0]
