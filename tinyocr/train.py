"""Train the recognizer on synthetic lines, or on a TSV of real crops."""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from tinyocr.charset import CHARSET, encode, normalize_label
from tinyocr.config import BLANK, IMG_H, IMG_W, TIME_STEPS
from tinyocr.decode import greedy_decode
from tinyocr.fonts import TRAIN_FILENAMES, resolve_fonts
from tinyocr.metrics import score_predictions
from tinyocr.model import TinyRecognizer
from tinyocr.preprocess import prepare_line
from tinyocr.synth import STANDARD, SynthSource


class ArrayLines(Dataset):
    def __init__(self, images: np.ndarray, texts: list[str]):
        self.images = images
        self.targets = [torch.tensor(encode(text), dtype=torch.long) for text in texts]
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int):
        image = torch.from_numpy(self.images[index])
        return image, self.targets[index]


class LabeledLines(Dataset):
    def __init__(self, rows: list[tuple[Path, str]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        path, text = self.rows[index]
        with Image.open(path) as image:
            tensor = prepare_line(image, IMG_H, IMG_W)
        return torch.from_numpy(tensor), torch.tensor(encode(text), dtype=torch.long)


def collate(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    lengths = torch.tensor([item[1].numel() for item in batch], dtype=torch.long)
    targets = torch.cat([item[1] for item in batch], dim=0)
    return images, targets, lengths


def read_labels(path: Path) -> list[tuple[Path, str]]:
    base = path.parent
    rows: list[tuple[Path, str]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "\t" not in stripped:
            raise SystemExit(f"{path}:{line_no}: expected image<TAB>text")
        rel, text = stripped.split("\t", 1)
        image_path = Path(rel)
        if not image_path.is_file():
            image_path = base / rel
        if not image_path.is_file():
            raise SystemExit(f"{path}:{line_no}: missing image {rel}")
        try:
            label = normalize_label(text)
        except ValueError as exc:
            raise SystemExit(f"{path}:{line_no}: {exc}") from exc
        if len(label) >= TIME_STEPS:
            raise SystemExit(
                f"{path}:{line_no}: label length {len(label)} needs a wider input "
                f"than {IMG_W}px ({TIME_STEPS} time steps)"
            )
        rows.append((image_path, label))
    if len(rows) < 2:
        raise SystemExit("need at least 2 labeled lines")
    return rows


def split_rows(rows: list[tuple[Path, str]], seed: int) -> tuple[list, list]:
    order = np.random.default_rng(seed).permutation(len(rows))
    shuffled = [rows[int(i)] for i in order]
    val_n = max(1, len(shuffled) // 10)
    return shuffled[val_n:], shuffled[:val_n]


@torch.no_grad()
def predict_loader(model: TinyRecognizer, loader: DataLoader, texts: list[str] | None) -> list[str]:
    model.eval()
    preds: list[str] = []
    for images, _, _ in loader:
        log_probs = model(images).numpy()
        preds.extend(greedy_decode(log_probs, CHARSET))
    if texts is not None and len(preds) != len(texts):
        raise RuntimeError("prediction count does not match labels")
    return preds


def run_epoch(
    model: TinyRecognizer,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    seen = 0
    for images, targets, target_lengths in loader:
        log_probs = model(images)
        time_steps = log_probs.shape[1]
        input_lengths = torch.full((images.shape[0],), time_steps, dtype=torch.long)
        loss = criterion(
            log_probs.transpose(0, 1),
            targets,
            input_lengths,
            target_lengths,
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        total += float(loss.item()) * images.shape[0]
        seen += images.shape[0]
    return total / max(seen, 1)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    epoch: int,
    metrics: dict,
    height: int = IMG_H,
    width: int = IMG_W,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "charset": CHARSET,
            "blank": BLANK,
            "height": height,
            "width": width,
            "epoch": epoch,
            "val_exact": metrics["exact_accuracy"],
            "val_character_accuracy": metrics["character_accuracy"],
        },
        path,
    )


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_init(model: nn.Module, path: Path, strict: bool = True) -> None:
    ckpt = load_checkpoint(path)
    if ckpt.get("charset") != CHARSET:
        raise SystemExit(f"{path} was trained with a different charset")
    model.load_state_dict(ckpt["model"], strict=strict)


def train(args: argparse.Namespace) -> int:
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    model = TinyRecognizer(len(CHARSET) + 1)
    if args.init is not None:
        load_init(model, args.init)
    params = sum(p.numel() for p in model.parameters())
    print(f"parameters {params}", flush=True)

    if args.labels is not None:
        rows = read_labels(args.labels)
        train_rows, val_rows = split_rows(rows, args.seed)
        train_ds: Dataset = LabeledLines(train_rows)
        val_ds: Dataset = LabeledLines(val_rows)
        val_texts = [text for _, text in val_rows]
        print(f"labeled train {len(train_ds)}  val {len(val_ds)}", flush=True)
    else:
        fonts = resolve_fonts(TRAIN_FILENAMES, args.font)
        if len(fonts) < 3:
            raise SystemExit(
                "found fewer than 3 training fonts. Install DejaVu or Liberation, "
                "or pass --font repeatedly."
            )
        print(f"training fonts {len(fonts)}", flush=True)
        source = SynthSource(fonts)
        started = time.perf_counter()
        train_images, train_texts = source.materialize(args.train_size, seed=10_000 + args.seed, augment=STANDARD)
        val_images, val_texts = source.materialize(args.val_size, seed=20_000 + args.seed, augment=STANDARD)
        print(
            f"rendered train {len(train_texts)} val {len(val_texts)} "
            f"in {time.perf_counter() - started:.1f}s",
            flush=True,
        )
        train_ds = ArrayLines(train_images, train_texts)
        val_ds = ArrayLines(val_images, val_texts)

    shuffle_seed = torch.Generator()
    shuffle_seed.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
        generator=shuffle_seed,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    criterion = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_exact = -1.0
    best_metrics: dict | None = None
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        loss = run_epoch(model, train_loader, criterion, optimizer)
        schedule.step()
        preview = predict_loader(model, val_loader, val_texts)
        metrics = score_predictions(preview, val_texts)
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch}  loss {loss:.4f}  "
            f"val_exact {metrics['exact_accuracy']:.4f}  "
            f"val_char {metrics['character_accuracy']:.4f}  lr {lr:.6f}",
            flush=True,
        )
        for truth, pred in list(zip(val_texts, preview))[:4]:
            mark = "ok" if truth == pred else "miss"
            print(f"  {mark}  {truth} -> {pred}", flush=True)
        if metrics["exact_accuracy"] > best_exact:
            best_exact = metrics["exact_accuracy"]
            best_metrics = metrics
            best_epoch = epoch
            save_checkpoint(args.out, model, epoch, metrics)
        if (
            epoch >= args.min_epochs
            and best_exact >= args.target_exact
            and epoch >= best_epoch
        ):
            # One extra epoch after the target so batch-norm stats are not from a single pass.
            if epoch > best_epoch:
                break

    if best_metrics is None:
        raise SystemExit("training produced no checkpoint")
    print(
        f"best epoch {best_epoch}  val_exact {best_metrics['exact_accuracy']:.4f}  "
        f"val_char {best_metrics['character_accuracy']:.4f}  saved {args.out}",
        flush=True,
    )
    print(
        "validation picks the checkpoint. Run `python -m tinyocr bench` for a fresh set.",
        flush=True,
    )
    if best_exact < args.target_exact:
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the tiny line recognizer")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--min-epochs", type=int, default=4)
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--val-size", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-exact", type=float, default=0.97)
    parser.add_argument("--out", type=Path, default=Path("artifacts/recognizer.pt"))
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--init", type=Path, default=None)
    parser.add_argument("--font", action="append", default=None)
    args = parser.parse_args(argv)
    return train(args)
