"""Synthetic printed lines. Training and evaluation share this renderer."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from tinyocr.charset import CHARSET, encode
from tinyocr.config import IMG_H, IMG_W, MAX_LEN, MIN_LEN
from tinyocr.preprocess import prepare_line

SAMPLE_TEXTS = ("7F2K", "A0B18", "MN94Q", "ZX831")


@dataclass(frozen=True)
class Augment:
    name: str
    max_noise: float
    blur_prob: float
    max_blur: float
    fg_max: int
    bg_min: int
    lowercase_prob: float


CLEAN = Augment("clean", 0.0, 0.0, 0.0, 0, 255, 0.0)
STANDARD = Augment("standard", 8.0, 0.4, 0.55, 30, 236, 0.5)
MESSY = Augment("messy", 16.0, 0.85, 1.1, 70, 210, 0.5)


def child_seed(seed: int, index: int) -> int:
    z = (seed + index * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    z = z ^ (z >> 31)
    return int(z)


def random_text(rng: np.random.Generator, min_len: int = MIN_LEN, max_len: int = MAX_LEN) -> str:
    length = int(rng.integers(min_len, max_len + 1))
    picks = rng.integers(0, len(CHARSET), size=length)
    return "".join(CHARSET[int(i)] for i in picks)


def _shown_text(text: str, rng: np.random.Generator, augment: Augment) -> str:
    if augment.lowercase_prob <= 0 or float(rng.random()) >= augment.lowercase_prob:
        return text
    return "".join(ch.lower() if ch.isalpha() else ch for ch in text)


def render_text(
    text: str,
    font: ImageFont.ImageFont,
    rng: np.random.Generator,
    augment: Augment,
) -> Image.Image:
    probe = Image.new("L", (1, 1), 255)
    bbox = ImageDraw.Draw(probe).textbbox((0, 0), text, font=font)
    tw = max(1, bbox[2] - bbox[0])
    th = max(1, bbox[3] - bbox[1])
    pad_left = int(rng.integers(2, 12))
    pad_right = int(rng.integers(4, 18))
    pad_y = int(rng.integers(1, 6))
    if augment.fg_max <= 0:
        fg = 0
    else:
        fg = int(rng.integers(0, augment.fg_max + 1))
    if augment.bg_min >= 255:
        bg = 255
    else:
        bg = int(rng.integers(augment.bg_min, 256))
    width = tw + pad_left + pad_right
    height = th + pad_y * 2
    image = Image.new("L", (width, height), color=bg)
    x = pad_left - bbox[0]
    y = pad_y - bbox[1]
    ImageDraw.Draw(image).text((x, y), text, font=font, fill=fg)
    if augment.blur_prob > 0 and float(rng.random()) < augment.blur_prob:
        radius = float(rng.uniform(0.15, augment.max_blur))
        image = image.filter(ImageFilter.GaussianBlur(radius=radius))
    if augment.max_noise > 0:
        sigma = float(rng.uniform(0.0, augment.max_noise))
        if sigma > 0:
            arr = np.asarray(image, dtype=np.float32)
            arr += rng.normal(0.0, sigma, size=arr.shape).astype(np.float32)
            image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")
    return image


class SynthSource:
    def __init__(self, font_paths: list[str], sizes: tuple[int, ...] = (20, 22, 24, 26, 28)):
        if not font_paths:
            raise RuntimeError("no font files available")
        self.font_paths = list(font_paths)
        self.faces: list[ImageFont.FreeTypeFont] = []
        for path in self.font_paths:
            for size in sizes:
                self.faces.append(ImageFont.truetype(path, size=size))

    def sample(
        self,
        index: int,
        seed: int,
        augment: Augment,
        min_len: int = MIN_LEN,
        max_len: int = MAX_LEN,
    ) -> tuple[np.ndarray, str]:
        rng = np.random.default_rng(child_seed(seed, index))
        text = random_text(rng, min_len, max_len)
        face = self.faces[int(rng.integers(0, len(self.faces)))]
        shown = _shown_text(text, rng, augment)
        image = render_text(shown, face, rng, augment)
        tensor = prepare_line(image, IMG_H, IMG_W)
        return tensor, text

    def materialize(
        self,
        count: int,
        seed: int,
        augment: Augment,
    ) -> tuple[np.ndarray, list[str]]:
        images = np.empty((count, 1, IMG_H, IMG_W), dtype=np.float32)
        texts: list[str] = []
        for i in range(count):
            tensor, text = self.sample(i, seed, augment)
            images[i] = tensor
            texts.append(text)
            if (i + 1) % 5000 == 0 or i + 1 == count:
                print(f"  rendered {i + 1}/{count}", flush=True)
        return images, texts


def write_samples(out_dir: Path, font_path: str) -> list[dict[str, str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    face = ImageFont.truetype(font_path, size=26)
    manifest: list[dict[str, str]] = []
    for index, text in enumerate(SAMPLE_TEXTS):
        encode(text)
        rng = np.random.default_rng(child_seed(5000, index))
        image = render_text(text, face, rng, CLEAN)
        dest = out_dir / f"{text}.png"
        image.save(dest)
        manifest.append({"file": dest.as_posix(), "text": text})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
