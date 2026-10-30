"""Fine-tune the recognizer on public hand-printed character datasets.

EMNIST Digits and EMNIST Letters come from NIST Special Database 19
(Cohen, Afshar, Tapson, van Schaik, 2017). Chars74k EnglishHnd is the
tablet set from de Campos, Babu, and Varma, VISAPP 2009. Both are
isolated characters. This module pastes them into short lines, which is
hand-printed text, not connected cursive.
"""

import argparse
import gzip
import os
import struct
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from tinyocr.charset import CHARSET
from tinyocr.config import IMG_H, IMG_W, MAX_LEN, MIN_LEN
from tinyocr.preprocess import prepare_line
from tinyocr.synth import child_seed, random_text

EMNIST_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
CHARS74K_URL = "https://info-ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishHnd.tgz"
CHARS74K_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
# Writers 46-55 stay out of training. Chars74k has 55 writers, numbered 1-55.
HELDOUT_WRITER_MIN = 46
# Glyphs that dominate the remaining handwriting errors.
CONFUSABLE = "O0I1LQ9"
CONFUSABLE_GROUPS = ("O0", "I1L", "Q9")

_EMNIST_FILES = (
    "emnist-digits-mapping.txt",
    "emnist-digits-train-images-idx3-ubyte.gz",
    "emnist-digits-train-labels-idx1-ubyte.gz",
    "emnist-digits-test-images-idx3-ubyte.gz",
    "emnist-digits-test-labels-idx1-ubyte.gz",
    "emnist-letters-mapping.txt",
    "emnist-letters-train-images-idx3-ubyte.gz",
    "emnist-letters-train-labels-idx1-ubyte.gz",
    "emnist-letters-test-images-idx3-ubyte.gz",
    "emnist-letters-test-labels-idx1-ubyte.gz",
)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    print(f"downloading {url}", flush=True)
    urllib.request.urlretrieve(url, partial)
    partial.replace(dest)


def _safe_members(archive: tarfile.TarFile, dest: Path):
    root = dest.resolve()
    for member in archive.getmembers():
        target = (root / member.name).resolve()
        if root != target and root not in target.parents:
            raise SystemExit(f"refusing to extract {member.name}")
        yield member


def ensure_data(data_dir: Path) -> None:
    gzip_dir = data_dir / "gzip"
    missing = [name for name in _EMNIST_FILES if not (gzip_dir / name).is_file()]
    if missing:
        archive = data_dir / "emnist-gzip.zip"
        if not archive.is_file():
            _download(EMNIST_URL, archive)
        wanted = set(_EMNIST_FILES)
        with zipfile.ZipFile(archive) as package:
            for info in package.infolist():
                if Path(info.filename).name not in wanted:
                    continue
                target = gzip_dir / Path(info.filename).name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(package.read(info))
    img_root = data_dir / "English" / "Hnd" / "Img"
    if not img_root.is_dir():
        archive = data_dir / "EnglishHnd.tgz"
        if not archive.is_file():
            _download(CHARS74K_URL, archive)
        with tarfile.open(archive) as package:
            package.extractall(data_dir, members=_safe_members(package, data_dir))
    if not img_root.is_dir():
        raise SystemExit(f"Chars74k images not found at {img_root}")


def read_idx_images(path: Path) -> np.ndarray:
    """Read an IDX image file. EMNIST stores each glyph transposed."""
    with gzip.open(path, "rb") as handle:
        data = handle.read()
    magic, count, rows, cols = struct.unpack(">IIII", data[:16])
    if magic != 2051:
        raise ValueError(f"{path} is not an IDX image file (magic {magic})")
    images = np.frombuffer(data, dtype=np.uint8, offset=16, count=count * rows * cols)
    images = images.reshape(count, rows, cols)
    return np.ascontiguousarray(images.transpose(0, 2, 1))


def read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        data = handle.read()
    magic, count = struct.unpack(">II", data[:8])
    if magic != 2049:
        raise ValueError(f"{path} is not an IDX label file (magic {magic})")
    return np.frombuffer(data, dtype=np.uint8, offset=8, count=count).copy()


def read_mapping(path: Path) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        mapping[int(parts[0])] = chr(int(parts[1])).upper()
    return mapping


class IndexedGlyphs:
    """Fixed-size EMNIST glyphs, white ink on black, indexed by character."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, mapping: dict[int, str]):
        if len(images) != len(labels):
            raise ValueError("image and label counts differ")
        self.images = images
        chars = np.array([mapping[int(label)] for label in labels])
        self.by_char = {
            char: np.flatnonzero(chars == char).astype(np.int32) for char in CHARSET
        }
        empty = [char for char, idxs in self.by_char.items() if idxs.size == 0]
        if empty:
            raise RuntimeError(f"glyph bank is missing {''.join(empty)}")

    def sample(self, char: str, rng: np.random.Generator) -> np.ndarray:
        pool = self.by_char[char]
        index = int(pool[int(rng.integers(0, pool.size))])
        # EMNIST ink is white. The line renderer wants dark ink on white.
        return crop_ink(255 - self.images[index])


class ListedGlyphs:
    def __init__(self, by_char: dict[str, list[np.ndarray]]):
        self.by_char = by_char

    def sample(self, char: str, rng: np.random.Generator) -> np.ndarray:
        pool = self.by_char[char]
        return pool[int(rng.integers(0, len(pool)))]

    def has(self, char: str) -> bool:
        return bool(self.by_char.get(char))


class MixedGlyphs:
    def __init__(self, primary: IndexedGlyphs, extra: ListedGlyphs, extra_prob: float):
        self.primary = primary
        self.extra = extra
        self.extra_prob = extra_prob

    def sample(self, char: str, rng: np.random.Generator) -> np.ndarray:
        if self.extra.has(char) and float(rng.random()) < self.extra_prob:
            return self.extra.sample(char, rng)
        return self.primary.sample(char, rng)


def crop_ink(image: np.ndarray, threshold: int = 230) -> np.ndarray:
    ink = np.nonzero(image < threshold)
    if ink[0].size == 0:
        return image
    y0, y1 = int(ink[0].min()), int(ink[0].max()) + 1
    x0, x1 = int(ink[1].min()), int(ink[1].max()) + 1
    return np.ascontiguousarray(image[y0:y1, x0:x1])


def _load_emnist_split(
    gzip_dir: Path, kind: str, split: str
) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
    images = read_idx_images(gzip_dir / f"emnist-{kind}-{split}-images-idx3-ubyte.gz")
    labels = read_idx_labels(gzip_dir / f"emnist-{kind}-{split}-labels-idx1-ubyte.gz")
    mapping = read_mapping(gzip_dir / f"emnist-{kind}-mapping.txt")
    return images, labels, mapping


def load_emnist(gzip_dir: Path, split: str) -> IndexedGlyphs:
    chunks = [_load_emnist_split(gzip_dir, kind, split) for kind in ("digits", "letters")]
    images = np.concatenate([chunk[0] for chunk in chunks], axis=0)
    # Digit labels are 0-9 and letter labels are 1-26, so letters are offset.
    mapping: dict[int, str] = {}
    offset = 0
    relabeled = []
    for _images, labels_part, part_map in chunks:
        shifted = {label + offset: char for label, char in part_map.items()}
        mapping.update(shifted)
        relabeled.append(labels_part.astype(np.int32) + offset)
        offset += max(part_map) + 1
    return IndexedGlyphs(images, np.concatenate(relabeled), mapping)


def _limit_height(glyph: np.ndarray, max_height: int) -> np.ndarray:
    """Cap a glyph's height. The default cap keeps the 32px benches unchanged."""
    if glyph.shape[0] <= max_height:
        return np.ascontiguousarray(glyph)
    new_w = max(1, int(round(glyph.shape[1] * (max_height / glyph.shape[0]))))
    resized = np.asarray(
        Image.fromarray(glyph).resize((new_w, max_height), Image.Resampling.BILINEAR)
    )
    return np.ascontiguousarray(resized)


def load_chars74k(
    img_root: Path, writer_min: int, writer_max: int, max_height: int = 48
) -> ListedGlyphs:
    by_char: dict[str, list[np.ndarray]] = {char: [] for char in CHARSET}
    for class_dir in sorted(img_root.glob("Sample*")):
        class_index = int(class_dir.name.replace("Sample", "")) - 1
        if class_index < 0 or class_index >= len(CHARS74K_ALPHABET):
            continue
        char = CHARS74K_ALPHABET[class_index].upper()
        for path in class_dir.glob("*.png"):
            writer = int(path.stem.split("-")[-1])
            if writer < writer_min or writer > writer_max:
                continue
            gray = np.asarray(Image.open(path).convert("L"))
            if float(gray.mean()) < 127:
                gray = 255 - gray
            glyph = _limit_height(crop_ink(gray), max_height)
            by_char[char].append(glyph)
    missing = [char for char, glyphs in by_char.items() if not glyphs]
    if missing:
        raise RuntimeError(f"Chars74k is missing {''.join(missing)} for writers {writer_min}-{writer_max}")
    return ListedGlyphs(by_char)


def blit_min(canvas: np.ndarray, glyph: np.ndarray, x: int, y: int) -> None:
    height, width = glyph.shape
    src_y0 = max(0, -y)
    src_x0 = max(0, -x)
    dst_y0 = max(0, y)
    dst_x0 = max(0, x)
    dst_y1 = min(canvas.shape[0], y + height)
    dst_x1 = min(canvas.shape[1], x + width)
    if dst_y0 >= dst_y1 or dst_x0 >= dst_x1:
        return
    src = glyph[src_y0 : src_y0 + (dst_y1 - dst_y0), src_x0 : src_x0 + (dst_x1 - dst_x0)]
    region = canvas[dst_y0:dst_y1, dst_x0:dst_x1]
    np.minimum(region, src, out=region)


def _fit_glyph(
    glyph: np.ndarray,
    rng: np.random.Generator,
    height_low: int = 20,
    height_high: int = 32,
) -> np.ndarray:
    image = Image.fromarray(glyph, mode="L")
    target_h = int(rng.integers(height_low, height_high))
    target_w = max(1, int(round(image.width * (target_h / max(image.height, 1)))))
    image = image.resize((target_w, target_h), Image.Resampling.BILINEAR)
    angle = float(rng.uniform(-8.0, 8.0))
    image = image.rotate(angle, expand=True, resample=Image.Resampling.BILINEAR, fillcolor=255)
    return np.asarray(image)


def confusable_line(
    rng: np.random.Generator, min_len: int = MIN_LEN, max_len: int = MAX_LEN
) -> str:
    """A random line with at least two glyphs from the confusable set, then shuffled."""
    length = int(rng.integers(min_len, max_len + 1))
    chars = [CHARSET[int(index)] for index in rng.integers(0, len(CHARSET), size=length)]
    for slot in rng.choice(length, size=2, replace=False):
        chars[int(slot)] = CONFUSABLE[int(rng.integers(0, len(CONFUSABLE)))]
    rng.shuffle(chars)
    return "".join(chars)


def ctc_sample_weights(targets: list[int], lengths: list[int]) -> list[float]:
    """Give confusable labels twice the CTC weight. Other labels stay at 1."""
    confusable_ids = {CHARSET.index(char) + 1 for char in CONFUSABLE}
    weights: list[float] = []
    offset = 0
    for length in lengths:
        chunk = targets[offset : offset + length]
        offset += length
        weights.append(2.0 if any(index in confusable_ids for index in chunk) else 1.0)
    return weights


def render_hand_line(
    bank,
    rng: np.random.Generator,
    text: str | None = None,
    confusable_rate: float = 0.0,
    stroke_prob: float = 0.0,
    glyph_low: int = 20,
    glyph_high: int = 32,
    canvas_height: int = 52,
    baseline: int = 40,
) -> tuple[Image.Image, str]:
    if text is None:
        # A zero rate must not draw from the rng, so the bench lines stay fixed.
        if confusable_rate > 0 and float(rng.random()) < confusable_rate:
            text = confusable_line(rng)
        else:
            text = random_text(rng, MIN_LEN, MAX_LEN)
    glyphs = [
        _fit_glyph(bank.sample(char, rng), rng, glyph_low, glyph_high) for char in text
    ]
    gaps = [int(rng.integers(-1, 5)) for _ in glyphs]
    width = 12 + sum(glyph.shape[1] for glyph in glyphs) + sum(gaps)
    width = max(width, 16)
    canvas = np.full((canvas_height, width), 255, dtype=np.uint8)
    x = 4
    for glyph, gap in zip(glyphs, gaps):
        y = baseline - glyph.shape[0] + int(rng.integers(-2, 3))
        blit_min(canvas, glyph, x, y)
        x += glyph.shape[1] + gap
    image = Image.fromarray(canvas, mode="L")
    if stroke_prob > 0 and float(rng.random()) < stroke_prob:
        # MinFilter spreads dark ink by one pixel. MaxFilter thins it.
        if float(rng.random()) < 0.5:
            image = image.filter(ImageFilter.MinFilter(3))
        else:
            image = image.filter(ImageFilter.MaxFilter(3))
    shear = float(rng.uniform(-0.15, 0.15))
    image = image.transform(
        image.size,
        Image.Transform.AFFINE,
        (1, shear, 0, 0, 1, 0),
        resample=Image.Resampling.BILINEAR,
        fillcolor=255,
    )
    if float(rng.random()) < 0.25:
        image = image.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.15, 0.45))))
    arr = np.asarray(image, dtype=np.float32)
    sigma = float(rng.uniform(0.0, 5.0))
    if sigma > 0:
        arr += rng.normal(0.0, sigma, size=arr.shape).astype(np.float32)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L"), text


def materialize(
    bank,
    count: int,
    seed: int,
    confusable_rate: float = 0.0,
    stroke_prob: float = 0.0,
    height: int = IMG_H,
    width: int = IMG_W,
    glyph_low: int = 20,
    glyph_high: int = 32,
    canvas_height: int = 52,
    baseline: int = 40,
) -> tuple[np.ndarray, list[str]]:
    images = np.empty((count, 1, height, width), dtype=np.float32)
    texts: list[str] = []
    for index in range(count):
        rng = np.random.default_rng(child_seed(seed, index))
        image, text = render_hand_line(
            bank,
            rng,
            confusable_rate=confusable_rate,
            stroke_prob=stroke_prob,
            glyph_low=glyph_low,
            glyph_high=glyph_high,
            canvas_height=canvas_height,
            baseline=baseline,
        )
        images[index] = prepare_line(image, height, width)
        texts.append(text)
        if (index + 1) % 5000 == 0 or index + 1 == count:
            print(f"  rendered {index + 1}/{count}", flush=True)
    return images, texts


def _weighted_epoch(model, loader, criterion, optimizer) -> float:
    """CTC loss with a higher weight on lines that contain a confusable glyph."""
    import torch

    model.train()
    total = 0.0
    seen = 0
    for images, targets, target_lengths in loader:
        log_probs = model(images)
        time_steps = log_probs.shape[1]
        input_lengths = torch.full((images.shape[0],), time_steps, dtype=torch.long)
        losses = criterion(
            log_probs.transpose(0, 1),
            targets,
            input_lengths,
            target_lengths,
        )
        weights = torch.tensor(
            ctc_sample_weights(targets.tolist(), target_lengths.tolist()),
            dtype=losses.dtype,
        )
        # Match CTCLoss(reduction="mean"), which divides each sample by its length.
        per_label = losses / target_lengths.to(dtype=losses.dtype).clamp_min(1)
        loss = (per_label * weights).sum() / weights.sum().clamp_min(1.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += float(loss.item()) * images.shape[0]
        seen += images.shape[0]
    return total / max(seen, 1)


def _group_overlap(vectors: np.ndarray, labels: np.ndarray) -> dict:
    """Leave-one-out L2 neighbor. Rival means the neighbor has another label."""
    norms = np.sum(vectors * vectors, axis=1)
    distances = norms[:, None] + norms[None, :] - 2.0 * (vectors @ vectors.T)
    np.fill_diagonal(distances, np.inf)
    nearest = np.argmin(distances, axis=1)
    rival = labels[nearest] != labels
    classes: dict[str, dict] = {}
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        classes[label] = {
            "n": int(mask.sum()),
            "rival_neighbor_rate": float(rival[mask].mean()) if mask.any() else 0.0,
        }
    return {
        "n": int(len(labels)),
        "rival_neighbor_rate": float(rival.mean()) if len(labels) else 0.0,
        "classes": classes,
    }


def write_overlap(args: argparse.Namespace) -> int:
    """Pixel-space overlap of confusable EMNIST glyphs. Not a Bayes ceiling."""
    import json

    ensure_data(args.data_dir)
    bank = load_emnist(args.data_dir / "gzip", "test")
    rng = np.random.default_rng(args.seed)
    groups = {}
    for name in CONFUSABLE_GROUPS:
        chosen_images = []
        chosen_labels = []
        for char in name:
            pool = bank.by_char[char]
            take = min(args.per_class, int(pool.size))
            picked = rng.choice(pool, size=take, replace=False)
            for index in picked:
                # Inverted 28x28: dark ink on a light field, same size for every glyph.
                glyph = (255 - bank.images[int(index)]).astype(np.float32).reshape(-1)
                chosen_images.append(glyph)
                chosen_labels.append(char)
        vectors = np.stack(chosen_images).astype(np.float32)
        groups[name] = _group_overlap(vectors, np.array(chosen_labels))
    report = {
        "kind": "pixel nearest-neighbor overlap",
        "note": (
            "Share of EMNIST test glyphs whose nearest other glyph in the same "
            "confusable group has a different label. This is overlap in 28x28 "
            "pixel space, not a Bayes error ceiling."
        ),
        "space": "28x28 inverted EMNIST, L2, leave-one-out inside each group",
        "per_class": args.per_class,
        "seed": args.seed,
        "groups": groups,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, group in groups.items():
        print(
            f"{name}  rival_neighbor {group['rival_neighbor_rate']:.4f}  n {group['n']}",
            flush=True,
        )
    print(f"wrote {args.out}", flush=True)
    return 0


def train_hand(args: argparse.Namespace) -> int:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from tinyocr.fonts import TRAIN_FILENAMES, resolve_fonts
    from tinyocr.metrics import score_predictions
    from tinyocr.model import TinyRecognizer
    from tinyocr.synth import STANDARD, SynthSource
    from tinyocr.train import ArrayLines, collate, load_init, predict_loader, run_epoch, save_checkpoint

    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    ensure_data(args.data_dir)
    print("loading EMNIST train split", flush=True)
    emnist_train = load_emnist(args.data_dir / "gzip", "train")
    chars_train = load_chars74k(
        args.data_dir / "English" / "Hnd" / "Img", writer_min=1, writer_max=HELDOUT_WRITER_MIN - 1
    )
    print("loading EMNIST test split", flush=True)
    emnist_test = load_emnist(args.data_dir / "gzip", "test")
    train_bank = MixedGlyphs(emnist_train, chars_train, extra_prob=0.2)
    hand_count = args.train_size - int(args.train_size * args.replay)
    confusable_rate = float(getattr(args, "confusable_rate", 0.0))
    stroke_prob = float(getattr(args, "stroke_prob", 0.0))
    print(f"rendering {hand_count} handwriting lines", flush=True)
    images, texts = materialize(
        train_bank,
        hand_count,
        seed=30_000 + args.seed,
        confusable_rate=confusable_rate,
        stroke_prob=stroke_prob,
    )
    if args.replay > 0:
        fonts = resolve_fonts(TRAIN_FILENAMES, None)
        if len(fonts) < 3:
            raise SystemExit("printed replay needs the training fonts, or pass --replay 0")
        printed_count = args.train_size - hand_count
        print(f"rendering {printed_count} printed replay lines", flush=True)
        printed_images, printed_texts = SynthSource(fonts).materialize(
            printed_count, seed=31_000 + args.seed, augment=STANDARD
        )
        images = np.concatenate([images, printed_images], axis=0)
        texts = texts + printed_texts
        order = np.random.default_rng(args.seed).permutation(len(texts))
        images = images[order]
        texts = [texts[int(i)] for i in order]
    print(f"rendering {args.val_size} EMNIST test lines", flush=True)
    val_images, val_texts = materialize(emnist_test, args.val_size, seed=40_000 + args.seed)

    model = TinyRecognizer(len(CHARSET) + 1)
    if args.init is not None:
        load_init(model, args.init)
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
    weighted = confusable_rate > 0
    criterion = nn.CTCLoss(
        blank=0, zero_infinity=True, reduction="none" if weighted else "mean"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    best_exact = -1.0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        if weighted:
            loss = _weighted_epoch(model, train_loader, criterion, optimizer)
        else:
            loss = run_epoch(model, train_loader, criterion, optimizer)
        schedule.step()
        preds = predict_loader(model, val_loader, val_texts)
        metrics = score_predictions(preds, val_texts)
        print(
            f"epoch {epoch}  loss {loss:.4f}  "
            f"hand_exact {metrics['exact_accuracy']:.4f}  "
            f"hand_char {metrics['character_accuracy']:.4f}",
            flush=True,
        )
        for truth, pred in list(zip(val_texts, preds))[:4]:
            mark = "ok" if truth == pred else "miss"
            print(f"  {mark}  {truth} -> {pred}", flush=True)
        if metrics["exact_accuracy"] > best_exact:
            best_exact = metrics["exact_accuracy"]
            best_epoch = epoch
            save_checkpoint(args.out, model, epoch, metrics)
    print(f"best epoch {best_epoch}  hand_exact {best_exact:.4f}  saved {args.out}", flush=True)
    from tinyocr.export_onnx import export_checkpoint

    export_checkpoint(args.out, args.out.with_suffix(".onnx"))
    return 0


def bench_hand(args: argparse.Namespace) -> int:
    import json

    from tinyocr.bench import _latency_ms
    from tinyocr.decode import greedy_decode
    from tinyocr.fonts import TRAIN_FILENAMES, resolve_fonts
    from tinyocr.metrics import score_predictions
    from tinyocr.session import load_session, load_spec, predict_batch
    from tinyocr.synth import STANDARD, SynthSource

    ensure_data(args.data_dir)
    spec = load_spec(args.model)
    session = load_session(args.model, threads=4)
    probe = np.random.default_rng(0).random((8, 1, spec["height"], spec["width"])).astype(np.float32)
    latency = _latency_ms(session, spec, probe, args.repeats)
    latency["threads"] = 4
    print(
        f"model p50 {latency['p50_ms']:.3f} ms  p95 {latency['p95_ms']:.3f} ms",
        flush=True,
    )

    def score_bank(bank, count: int, seed: int) -> dict:
        images, texts = materialize(bank, count, seed=seed)
        preds: list[str] = []
        for start in range(0, count, 64):
            log_probs = predict_batch(session, spec, images[start : start + 64])
            preds.extend(greedy_decode(log_probs, spec["charset"]))
        return score_predictions(preds, texts)

    print("loading EMNIST test split", flush=True)
    emnist_test = load_emnist(args.data_dir / "gzip", "test")
    report = {
        "model": args.model.as_posix(),
        "latency": latency,
        "emnist_lines": score_bank(emnist_test, args.n, seed=args.seed),
    }
    report["emnist_lines"]["source"] = "EMNIST digits+letters official test characters, composed into lines"
    chars_test = load_chars74k(
        args.data_dir / "English" / "Hnd" / "Img",
        writer_min=HELDOUT_WRITER_MIN,
        writer_max=55,
    )
    report["chars74k_heldout_writers"] = score_bank(chars_test, min(args.n, 1000), seed=args.seed + 1)
    report["chars74k_heldout_writers"]["writers"] = f"{HELDOUT_WRITER_MIN}-55"
    fonts = resolve_fonts(TRAIN_FILENAMES, None)
    if len(fonts) >= 3:
        images, texts = SynthSource(fonts).materialize(min(args.n, 1000), seed=args.seed + 2, augment=STANDARD)
        preds = []
        for start in range(0, len(texts), 64):
            log_probs = predict_batch(session, spec, images[start : start + 64])
            preds.extend(greedy_decode(log_probs, spec["charset"]))
        report["printed_replay_check"] = score_predictions(preds, texts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name in ("emnist_lines", "chars74k_heldout_writers", "printed_replay_check"):
        block = report.get(name, {})
        if "exact_accuracy" in block:
            print(
                f"{name}  exact {block['exact_accuracy']:.4f}  "
                f"char {block['character_accuracy']:.4f}  n {block['n']}",
                flush=True,
            )
    print(f"wrote {args.out}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune on handwriting lines")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-size", type=int, default=32000)
    parser.add_argument("--val-size", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--replay", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init", type=Path, default=Path("artifacts/recognizer.pt"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/handwriting.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/handwriting"))
    return train_hand(parser.parse_args(argv))
