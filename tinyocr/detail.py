"""Height-64 recognizer trained on sharp Chars74k lines.

The shipped 32px graphs stay as they are. This sibling adds one height
pool so a 64px line still collapses to one row and 40 time steps. EMNIST
glyphs are native 28px, so this trainer uses Chars74k at a higher cap.
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

from tinyocr.config import DETAIL_H, DETAIL_W, IMG_H, IMG_W
from tinyocr.handwriting import (
    HELDOUT_WRITER_MIN,
    ensure_data,
    load_chars74k,
    render_hand_line,
)
from tinyocr.preprocess import prepare_line
from tinyocr.synth import child_seed

# rng.integers treats the high bound as exclusive, so 65 includes 64px.
DETAIL_GLYPH_LOW = 40
DETAIL_GLYPH_HIGH = 65
DETAIL_CANVAS_H = 112
DETAIL_BASELINE = 96
DETAIL_MAX_HEIGHT = 128
DETAIL_FONT_SIZES = (40, 44, 48, 52, 56)

# Shipped 32px Chars74k writers 46-55 character accuracy is 0.8766.
SHIPPED_CHARS_CHAR = 0.8766
GATE_CHARS_CHAR = SHIPPED_CHARS_CHAR + 0.02
GATE_PRINTED_EXACT = 0.94
GATE_P50_MS = 1.0
PAIRED_GAIN = 0.02

_GEOMETRY = dict(
    glyph_low=DETAIL_GLYPH_LOW,
    glyph_high=DETAIL_GLYPH_HIGH,
    canvas_height=DETAIL_CANVAS_H,
    baseline=DETAIL_BASELINE,
)


def _stack(images: list[Image.Image], height: int, width: int) -> np.ndarray:
    batch = np.empty((len(images), 1, height, width), dtype=np.float32)
    for index, image in enumerate(images):
        batch[index] = prepare_line(image, height, width)
    return batch


def render_detail_lines(
    bank,
    count: int,
    seed: int,
    confusable_rate: float = 0.0,
    stroke_prob: float = 0.0,
) -> tuple[list[Image.Image], list[str]]:
    images: list[Image.Image] = []
    texts: list[str] = []
    for index in range(count):
        rng = np.random.default_rng(child_seed(seed, index))
        image, text = render_hand_line(
            bank,
            rng,
            confusable_rate=confusable_rate,
            stroke_prob=stroke_prob,
            **_GEOMETRY,
        )
        images.append(image)
        texts.append(text)
        if (index + 1) % 5000 == 0 or index + 1 == count:
            print(f"  rendered {index + 1}/{count}", flush=True)
    return images, texts


def _score_batch(session, spec: dict, images: np.ndarray, texts: list[str]) -> dict:
    from tinyocr.decode import greedy_decode
    from tinyocr.metrics import score_predictions
    from tinyocr.session import predict_batch

    preds: list[str] = []
    for start in range(0, len(texts), 64):
        log_probs = predict_batch(session, spec, images[start : start + 64])
        preds.extend(greedy_decode(log_probs, spec["charset"]))
    return score_predictions(preds, texts)


def train_detail(args: argparse.Namespace) -> int:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from tinyocr.charset import CHARSET
    from tinyocr.fonts import TRAIN_FILENAMES, resolve_fonts
    from tinyocr.handwriting import _weighted_epoch, materialize
    from tinyocr.metrics import score_predictions
    from tinyocr.model import DetailRecognizer
    from tinyocr.synth import STANDARD, SynthSource
    from tinyocr.train import (
        ArrayLines,
        collate,
        load_init,
        predict_loader,
        run_epoch,
        save_checkpoint,
    )

    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    ensure_data(args.data_dir)
    img_root = args.data_dir / "English" / "Hnd" / "Img"
    print("loading Chars74k writers 1-45 at native resolution", flush=True)
    chars_train = load_chars74k(
        img_root,
        writer_min=1,
        writer_max=HELDOUT_WRITER_MIN - 1,
        max_height=DETAIL_MAX_HEIGHT,
    )
    hand_count = args.train_size - int(args.train_size * args.replay)
    print(f"rendering {hand_count} tall handwriting lines", flush=True)
    images, texts = materialize(
        chars_train,
        hand_count,
        seed=30_000 + args.seed,
        confusable_rate=args.confusable_rate,
        stroke_prob=args.stroke_prob,
        height=DETAIL_H,
        width=DETAIL_W,
        **_GEOMETRY,
    )
    fonts = resolve_fonts(TRAIN_FILENAMES, None)
    if len(fonts) < 3:
        raise SystemExit("detail training needs the training fonts for printed replay")
    printed_source = SynthSource(fonts, sizes=DETAIL_FONT_SIZES)
    if args.replay > 0:
        printed_count = args.train_size - hand_count
        print(f"rendering {printed_count} printed replay lines at {DETAIL_H}px", flush=True)
        printed_images, printed_texts = printed_source.materialize(
            printed_count,
            seed=31_000 + args.seed,
            augment=STANDARD,
            height=DETAIL_H,
            width=DETAIL_W,
        )
        images = np.concatenate([images, printed_images], axis=0)
        texts = texts + printed_texts
        order = np.random.default_rng(args.seed).permutation(len(texts))
        images = images[order]
        texts = [texts[int(i)] for i in order]

    print(f"rendering {args.val_size} held-out Chars74k lines", flush=True)
    chars_val = load_chars74k(
        img_root,
        writer_min=HELDOUT_WRITER_MIN,
        writer_max=55,
        max_height=DETAIL_MAX_HEIGHT,
    )
    val_images, val_texts = materialize(
        chars_val,
        args.val_size,
        seed=40_000 + args.seed,
        height=DETAIL_H,
        width=DETAIL_W,
        **_GEOMETRY,
    )
    print("rendering 1000 printed validation lines", flush=True)
    printed_val_images, printed_val_texts = printed_source.materialize(
        1000,
        seed=41_000 + args.seed,
        augment=STANDARD,
        height=DETAIL_H,
        width=DETAIL_W,
    )

    model = DetailRecognizer(len(CHARSET) + 1)
    if args.init is not None:
        load_init(model, args.init, strict=False)
        print(f"initialized from {args.init}", flush=True)
    params = sum(parameter.numel() for parameter in model.parameters())
    print(f"parameters {params}", flush=True)
    train_loader = DataLoader(
        ArrayLines(images, texts),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        ArrayLines(val_images, val_texts),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    printed_loader = DataLoader(
        ArrayLines(printed_val_images, printed_val_texts),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    weighted = args.confusable_rate > 0
    criterion = nn.CTCLoss(
        blank=0, zero_infinity=True, reduction="none" if weighted else "mean"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    best_key = (-1, -1.0, -1.0)
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        if weighted:
            loss = _weighted_epoch(model, train_loader, criterion, optimizer)
        else:
            loss = run_epoch(model, train_loader, criterion, optimizer)
        schedule.step()
        hand_preds = predict_loader(model, val_loader, val_texts)
        hand_metrics = score_predictions(hand_preds, val_texts)
        printed_preds = predict_loader(model, printed_loader, printed_val_texts)
        printed_metrics = score_predictions(printed_preds, printed_val_texts)
        eligible = 1 if printed_metrics["exact_accuracy"] >= GATE_PRINTED_EXACT else 0
        key = (
            eligible,
            hand_metrics["character_accuracy"],
            printed_metrics["exact_accuracy"],
        )
        print(
            f"epoch {epoch}  loss {loss:.4f}  "
            f"chars_exact {hand_metrics['exact_accuracy']:.4f}  "
            f"chars_char {hand_metrics['character_accuracy']:.4f}  "
            f"printed_exact {printed_metrics['exact_accuracy']:.4f}  "
            f"printed_char {printed_metrics['character_accuracy']:.4f}",
            flush=True,
        )
        for truth, pred in list(zip(val_texts, hand_preds))[:4]:
            mark = "ok" if truth == pred else "miss"
            print(f"  {mark}  {truth} -> {pred}", flush=True)
        if key > best_key:
            best_key = key
            best_epoch = epoch
            save_checkpoint(
                args.out, model, epoch, hand_metrics, height=DETAIL_H, width=DETAIL_W
            )
    print(
        f"best epoch {best_epoch}  chars_char {best_key[1]:.4f}  "
        f"printed_exact {best_key[2]:.4f}  saved {args.out}",
        flush=True,
    )
    from tinyocr.export_onnx import export_checkpoint

    export_checkpoint(args.out, args.out.with_suffix(".onnx"))
    return 0


def bench_detail(args: argparse.Namespace) -> int:
    import json

    from tinyocr.bench import _latency_ms
    from tinyocr.fonts import TRAIN_FILENAMES, resolve_fonts
    from tinyocr.session import load_session, load_spec
    from tinyocr.synth import STANDARD, SynthSource

    ensure_data(args.data_dir)
    spec = load_spec(args.model)
    if int(spec["height"]) != DETAIL_H or int(spec["width"]) != DETAIL_W:
        raise SystemExit(f"{args.model} is not a {DETAIL_H}x{DETAIL_W} detail model")
    session = load_session(args.model, threads=4)
    probe = np.random.default_rng(0).random((8, 1, DETAIL_H, DETAIL_W)).astype(np.float32)
    latency = _latency_ms(session, spec, probe, args.repeats)
    latency["threads"] = 4
    print(
        f"model p50 {latency['p50_ms']:.3f} ms  p95 {latency['p95_ms']:.3f} ms",
        flush=True,
    )

    img_root = args.data_dir / "English" / "Hnd" / "Img"
    print("loading Chars74k writers 46-55", flush=True)
    chars_test = load_chars74k(
        img_root,
        writer_min=HELDOUT_WRITER_MIN,
        writer_max=55,
        max_height=DETAIL_MAX_HEIGHT,
    )
    count = min(args.n, 1000)
    print(f"rendering {count} held-out lines", flush=True)
    hand_pils, hand_texts = render_detail_lines(chars_test, count, seed=args.seed + 1)
    tall = _stack(hand_pils, DETAIL_H, DETAIL_W)
    short = _stack(hand_pils, IMG_H, IMG_W)
    chars_detail = _score_batch(session, spec, tall, hand_texts)
    chars_detail["writers"] = f"{HELDOUT_WRITER_MIN}-55"
    chars_detail["render"] = "chars74k native cap 128, glyph height 40-64, input 64x160"

    paired = None
    if args.baseline.is_file():
        base_spec = load_spec(args.baseline)
        base_session = load_session(args.baseline, threads=4)
        paired = _score_batch(base_session, base_spec, short, hand_texts)
        paired["model"] = args.baseline.as_posix()
        paired["render"] = "same PIL lines prepared at 32x160"
        print(
            f"chars74k detail  exact {chars_detail['exact_accuracy']:.4f}  "
            f"char {chars_detail['character_accuracy']:.4f}",
            flush=True,
        )
        print(
            f"chars74k 32px    exact {paired['exact_accuracy']:.4f}  "
            f"char {paired['character_accuracy']:.4f}",
            flush=True,
        )

    printed_detail = None
    printed_paired = None
    printed_large = None
    fonts = resolve_fonts(TRAIN_FILENAMES, None)

    def score_printed(source: SynthSource, label: str) -> tuple[dict, dict | None]:
        printed_pils: list[Image.Image] = []
        printed_texts: list[str] = []
        printed_n = min(args.n, 1000)
        print(f"rendering {printed_n} {label} printed lines", flush=True)
        for index in range(printed_n):
            image, text = source.render_sample(index, args.seed + 2, STANDARD)
            printed_pils.append(image)
            printed_texts.append(text)
        detail_score = _score_batch(
            session, spec, _stack(printed_pils, DETAIL_H, DETAIL_W), printed_texts
        )
        print(
            f"{label} detail  exact {detail_score['exact_accuracy']:.4f}  "
            f"char {detail_score['character_accuracy']:.4f}",
            flush=True,
        )
        base_score = None
        if args.baseline.is_file():
            base_score = _score_batch(
                base_session,
                base_spec,
                _stack(printed_pils, IMG_H, IMG_W),
                printed_texts,
            )
            print(
                f"{label} 32px    exact {base_score['exact_accuracy']:.4f}  "
                f"char {base_score['character_accuracy']:.4f}",
                flush=True,
            )
        return detail_score, base_score

    if len(fonts) >= 3:
        # The 94% gate is the same printed distribution as the shipped bench.
        printed_detail, printed_paired = score_printed(SynthSource(fonts), "printed")
        printed_large, _printed_large_paired = score_printed(
            SynthSource(fonts, sizes=DETAIL_FONT_SIZES), "printed-large"
        )

    detail_char = float(chars_detail["character_accuracy"])
    paired_char = None if paired is None else float(paired["character_accuracy"])
    printed_exact = None if printed_detail is None else float(printed_detail["exact_accuracy"])
    reasons = {
        "chars_character_at_least_89.7": detail_char >= GATE_CHARS_CHAR,
        "chars_character_beats_same_ink_by_2_points": (
            paired_char is not None and detail_char >= paired_char + PAIRED_GAIN
        ),
        "printed_exact_at_least_94": (
            printed_exact is not None and printed_exact >= GATE_PRINTED_EXACT
        ),
        "p50_under_1ms": float(latency["p50_ms"]) < GATE_P50_MS,
    }
    accepted = all(reasons.values())
    report = {
        "model": args.model.as_posix(),
        "baseline": args.baseline.as_posix(),
        "latency": latency,
        "chars74k_heldout_writers": chars_detail,
        "chars74k_same_ink_32px": paired,
        "printed_replay_check": printed_detail,
        "printed_same_ink_32px": printed_paired,
        "printed_large_font": printed_large,
        "gates": {
            "chars_character": GATE_CHARS_CHAR,
            "paired_gain": PAIRED_GAIN,
            "printed_exact": GATE_PRINTED_EXACT,
            "p50_ms": GATE_P50_MS,
            "reasons": reasons,
            "accepted": accepted,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"accepted {accepted}", flush=True)
    for name, ok in reasons.items():
        print(f"  {name}  {ok}", flush=True)
    print(f"wrote {args.out}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the height-64 recognizer")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-size", type=int, default=14000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--replay", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--confusable-rate", type=float, default=0.4)
    parser.add_argument("--stroke-prob", type=float, default=0.5)
    parser.add_argument("--init", type=Path, default=Path("artifacts/handwriting.pt"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/detail.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/handwriting"))
    return train_detail(parser.parse_args(argv))
