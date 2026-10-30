"""Command line. Training imports stay inside the train/export commands."""

import argparse
import time
from pathlib import Path

from tinyocr.session import load_session, load_spec, predict_image


def _read(args: argparse.Namespace) -> int:
    from PIL import Image

    spec = load_spec(args.model)
    session = load_session(args.model, threads=args.threads)
    with Image.open(args.image) as image:
        for _ in range(10):
            predict_image(session, spec, image)
        started = time.perf_counter()
        text = predict_image(session, spec, image)
        elapsed = (time.perf_counter() - started) * 1000.0
    print(text)
    print(f"{elapsed:.3f} ms")
    return 0


def _samples(args: argparse.Namespace) -> int:
    from tinyocr.fonts import TRAIN_FILENAMES, resolve_fonts
    from tinyocr.synth import write_samples

    fonts = resolve_fonts(TRAIN_FILENAMES, None)
    if not fonts:
        raise SystemExit("no training font installed; cannot render samples")
    manifest = write_samples(args.out, fonts[0])
    for row in manifest:
        print(f"{row['text']}  {row['file']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tinyocr",
        description="Tiny ONNX recognizer for short printed lines",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train", help="Train on synthetic lines or a labeled TSV")
    export = sub.add_parser("export", help="Export a checkpoint to ONNX")
    bench = sub.add_parser("bench", help="Measure accuracy and millisecond latency")
    read = sub.add_parser("read", help="Read one line image")
    hand = sub.add_parser("train-hand", help="Fine-tune on EMNIST and Chars74k handwriting")
    hand_bench = sub.add_parser("bench-hand", help="Score the handwriting model on held-out glyphs")
    overlap = sub.add_parser("overlap", help="Measure pixel overlap of confusable EMNIST glyphs")
    samples = sub.add_parser("samples", help="Write a few demo line images")

    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--min-epochs", type=int, default=4)
    train.add_argument("--train-size", type=int, default=20000)
    train.add_argument("--val-size", type=int, default=2000)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--target-exact", type=float, default=0.97)
    train.add_argument("--out", type=Path, default=Path("artifacts/recognizer.pt"))
    train.add_argument("--labels", type=Path, default=None)
    train.add_argument("--init", type=Path, default=None)
    train.add_argument("--font", action="append", default=None)

    export.add_argument("--checkpoint", type=Path, default=Path("artifacts/recognizer.pt"))
    export.add_argument("--out", type=Path, default=Path("artifacts/recognizer.onnx"))

    bench.add_argument("--model", type=Path, default=Path("artifacts/recognizer.onnx"))
    bench.add_argument("--n", type=int, default=2000)
    bench.add_argument("--seed", type=int, default=2026)
    bench.add_argument("--repeats", type=int, default=300)
    bench.add_argument("--threads", type=int, default=1)
    bench.add_argument("--out", type=Path, default=Path("artifacts/metrics.json"))

    read.add_argument("image", type=Path)
    read.add_argument("--model", type=Path, default=Path("artifacts/recognizer.onnx"))
    read.add_argument(
        "--threads",
        type=int,
        default=0,
        help="ONNX intra-op threads. 0 uses up to 4.",
    )

    hand.add_argument("--epochs", type=int, default=8)
    hand.add_argument("--train-size", type=int, default=32000)
    hand.add_argument("--val-size", type=int, default=2000)
    hand.add_argument("--batch-size", type=int, default=128)
    hand.add_argument("--lr", type=float, default=5e-4)
    hand.add_argument("--replay", type=float, default=0.2)
    hand.add_argument("--seed", type=int, default=0)
    hand.add_argument("--confusable-rate", type=float, default=0.0)
    hand.add_argument("--stroke-prob", type=float, default=0.0)
    hand.add_argument("--init", type=Path, default=Path("artifacts/recognizer.pt"))
    hand.add_argument("--out", type=Path, default=Path("artifacts/handwriting.pt"))
    hand.add_argument("--data-dir", type=Path, default=Path("data/handwriting"))

    hand_bench.add_argument("--model", type=Path, default=Path("artifacts/handwriting.onnx"))
    hand_bench.add_argument("--n", type=int, default=2000)
    hand_bench.add_argument("--seed", type=int, default=2026)
    hand_bench.add_argument("--repeats", type=int, default=200)
    hand_bench.add_argument("--out", type=Path, default=Path("artifacts/handwriting-metrics.json"))
    hand_bench.add_argument("--data-dir", type=Path, default=Path("data/handwriting"))

    overlap.add_argument("--data-dir", type=Path, default=Path("data/handwriting"))
    overlap.add_argument("--per-class", type=int, default=300)
    overlap.add_argument("--seed", type=int, default=2026)
    overlap.add_argument("--out", type=Path, default=Path("artifacts/confusable-ceiling.json"))

    samples.add_argument("--out", type=Path, default=Path("samples"))

    args = parser.parse_args(argv)
    if args.cmd == "train":
        from tinyocr.train import train as run_train

        return run_train(args)
    if args.cmd == "export":
        from tinyocr.export_onnx import export_checkpoint

        export_checkpoint(args.checkpoint, args.out)
        return 0
    if args.cmd == "bench":
        from tinyocr.bench import bench as run_bench

        return run_bench(args)
    if args.cmd == "read":
        return _read(args)
    if args.cmd == "train-hand":
        from tinyocr.handwriting import train_hand

        return train_hand(args)
    if args.cmd == "bench-hand":
        from tinyocr.handwriting import bench_hand

        return bench_hand(args)
    if args.cmd == "overlap":
        from tinyocr.handwriting import write_overlap

        return write_overlap(args)
    if args.cmd == "samples":
        return _samples(args)
    raise SystemExit(f"unknown command {args.cmd}")
