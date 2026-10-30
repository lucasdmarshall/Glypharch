"""Accuracy on a fresh synthetic set, and single-image latency in milliseconds."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from tinyocr.decode import greedy_decode
from tinyocr.fonts import HELDOUT_FILENAMES, TRAIN_FILENAMES, resolve_fonts
from tinyocr.metrics import score_predictions
from tinyocr.session import load_session, load_spec, predict_batch, predict_image
from tinyocr.synth import MESSY, STANDARD, SynthSource


def _latency_ms(session, spec: dict, batch: np.ndarray, repeats: int) -> dict[str, float]:
    warmup = np.ascontiguousarray(batch[0:1])
    for _ in range(40):
        predict_batch(session, spec, warmup)
    samples: list[float] = []
    for i in range(repeats):
        image = batch[i % batch.shape[0] : i % batch.shape[0] + 1]
        started = time.perf_counter()
        predict_batch(session, spec, image)
        samples.append((time.perf_counter() - started) * 1000.0)
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
    }


def _score_split(session, spec: dict, source: SynthSource, count: int, seed: int, augment) -> dict:
    images, texts = source.materialize(count, seed=seed, augment=augment)
    preds: list[str] = []
    step = 64
    for start in range(0, count, step):
        log_probs = predict_batch(session, spec, images[start : start + step])
        preds.extend(greedy_decode(log_probs, spec["charset"]))
    scored = score_predictions(preds, texts)
    scored["augment"] = augment.name
    return scored


def bench(args: argparse.Namespace) -> int:
    spec = load_spec(args.model)
    report: dict = {
        "model": args.model.as_posix(),
        "input": [1, 1, spec["height"], spec["width"]],
        "charset": spec["charset"],
    }
    probe = np.random.default_rng(0).random((8, 1, spec["height"], spec["width"])).astype(np.float32)
    by_threads = {}
    sessions = {}
    for threads in (1, 4):
        sessions[threads] = load_session(args.model, threads=threads)
        measured = _latency_ms(sessions[threads], spec, probe, args.repeats)
        measured["threads"] = threads
        by_threads[str(threads)] = measured
        print(
            f"threads {threads}  p50 {measured['p50_ms']:.3f} ms  "
            f"p95 {measured['p95_ms']:.3f} ms  mean {measured['mean_ms']:.3f} ms",
            flush=True,
        )
    report["latency_by_threads"] = by_threads
    fastest = min(by_threads.values(), key=lambda item: item["p50_ms"])
    report["latency"] = fastest
    session = sessions[int(fastest["threads"])]

    e2e: list[float] = []
    sample_dir = Path("samples")
    sample_paths = sorted(sample_dir.glob("*.png")) if sample_dir.is_dir() else []
    if sample_paths:
        images = [Image.open(path) for path in sample_paths]
        for _ in range(10):
            for image in images:
                predict_image(session, spec, image)
        for _ in range(max(20, args.repeats // 4)):
            for image in images:
                started = time.perf_counter()
                predict_image(session, spec, image)
                e2e.append((time.perf_counter() - started) * 1000.0)
        for image in images:
            image.close()
        arr = np.asarray(e2e, dtype=np.float64)
        report["end_to_end_samples"] = {
            "n": int(arr.size),
            "mean_ms": float(arr.mean()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
        }

    train_fonts = resolve_fonts(TRAIN_FILENAMES, None)
    heldout_fonts = resolve_fonts(HELDOUT_FILENAMES, None)
    if len(train_fonts) >= 3:
        source = SynthSource(train_fonts)
        report["fresh_standard"] = _score_split(
            session, spec, source, args.n, seed=args.seed, augment=STANDARD
        )
        report["fresh_messy"] = _score_split(
            session, spec, source, min(args.n, 1000), seed=args.seed + 1, augment=MESSY
        )
    else:
        report["fresh_standard"] = {"skipped": "training fonts not installed"}
    if heldout_fonts:
        held = SynthSource(heldout_fonts)
        report["heldout_font"] = _score_split(
            session, spec, held, min(args.n, 1000), seed=args.seed + 2, augment=STANDARD
        )
        report["heldout_font"]["fonts"] = [Path(path).name for path in heldout_fonts]
    else:
        report["heldout_font"] = {"skipped": "held-out fonts not installed"}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    latency = report["latency"]
    print(
        f"model p50 {latency['p50_ms']:.3f} ms   p95 {latency['p95_ms']:.3f} ms   "
        f"mean {latency['mean_ms']:.3f} ms  (threads={latency['threads']})",
        flush=True,
    )
    fresh = report.get("fresh_standard", {})
    if "exact_accuracy" in fresh:
        print(
            f"fresh exact {fresh['exact_accuracy']:.4f}   "
            f"char {fresh['character_accuracy']:.4f}   n {fresh['n']}",
            flush=True,
        )
    messy = report.get("fresh_messy", {})
    if "exact_accuracy" in messy:
        print(
            f"messy exact {messy['exact_accuracy']:.4f}   "
            f"char {messy['character_accuracy']:.4f}   n {messy['n']}",
            flush=True,
        )
    held = report.get("heldout_font", {})
    if "exact_accuracy" in held:
        print(
            f"held-out font exact {held['exact_accuracy']:.4f}   "
            f"char {held['character_accuracy']:.4f}   n {held['n']}",
            flush=True,
        )
    print(f"wrote {args.out}", flush=True)
    if "exact_accuracy" in fresh and fresh["exact_accuracy"] < 0.97:
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark accuracy and latency")
    parser.add_argument("--model", type=Path, default=Path("artifacts/recognizer.onnx"))
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("artifacts/metrics.json"))
    args = parser.parse_args(argv)
    return bench(args)
