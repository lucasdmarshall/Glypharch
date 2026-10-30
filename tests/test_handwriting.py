import gzip
import struct
from pathlib import Path

import numpy as np

from PIL import Image

from tinyocr.charset import encode
from tinyocr.handwriting import (
    blit_min,
    confusable_line,
    crop_ink,
    ctc_sample_weights,
    read_idx_images,
    read_idx_labels,
    read_mapping,
    render_hand_line,
)


def _write_images(path: Path, images: np.ndarray) -> None:
    count, rows, cols = images.shape
    payload = struct.pack(">IIII", 2051, count, rows, cols) + np.ascontiguousarray(images).tobytes()
    with gzip.open(path, "wb") as handle:
        handle.write(payload)


def test_idx_transpose_restores_a_vertical_stroke(tmp_path: Path):
    stored = np.zeros((1, 4, 4), dtype=np.uint8)
    stored[0, 0, :] = 255
    path = tmp_path / "images.gz"
    _write_images(path, stored)
    loaded = read_idx_images(path)
    assert loaded.shape == (1, 4, 4)
    assert loaded[0, :, 0].tolist() == [255, 255, 255, 255]
    assert int(loaded[0, 0, 1]) == 0


def test_idx_labels(tmp_path: Path):
    path = tmp_path / "labels.gz"
    payload = struct.pack(">II", 2049, 3) + bytes([0, 1, 26])
    with gzip.open(path, "wb") as handle:
        handle.write(payload)
    assert read_idx_labels(path).tolist() == [0, 1, 26]


def test_mapping_uses_the_uppercase_ascii_column(tmp_path: Path):
    path = tmp_path / "map.txt"
    path.write_text("1 65 97\n0 48\n", encoding="utf-8")
    assert read_mapping(path) == {1: "A", 0: "0"}


def test_confusable_line_keeps_two_glyphs():
    rng = np.random.default_rng(0)
    for _ in range(40):
        text = confusable_line(rng)
        assert 4 <= len(text) <= 8
        assert sum(char in "O0I1LQ9" for char in text) >= 2


def test_ctc_weight_doubles_confusable_labels():
    plain = encode("AB")
    mixed = encode("A0")
    weights = ctc_sample_weights(plain + mixed, [len(plain), len(mixed)])
    assert weights == [1.0, 2.0]


def test_one_pixel_dilate_and_erode():
    from PIL import ImageFilter

    image = Image.new("L", (5, 5), 255)
    image.putpixel((2, 2), 0)
    spread = image.filter(ImageFilter.MinFilter(3))
    thinned = image.filter(ImageFilter.MaxFilter(3))
    assert spread.getpixel((2, 2)) == 0
    assert spread.getpixel((1, 2)) == 0
    assert thinned.getpixel((2, 2)) == 255


def test_render_accepts_a_forced_confusable_line():
    class _Bank:
        def sample(self, _char, _rng):
            glyph = np.full((10, 6), 255, dtype=np.uint8)
            glyph[2:8, 2:4] = 0
            return glyph

    image, text = render_hand_line(
        _Bank(),
        np.random.default_rng(3),
        text="O0",
        stroke_prob=1.0,
    )
    assert text == "O0"
    assert image.mode == "L"
    assert image.getextrema()[0] < 255


def test_crop_and_darker_ink_wins():
    image = np.full((6, 6), 255, dtype=np.uint8)
    image[2:5, 1:3] = 0
    assert crop_ink(image).shape == (3, 2)
    canvas = np.full((5, 5), 255, dtype=np.uint8)
    glyph = np.full((2, 2), 255, dtype=np.uint8)
    glyph[0, 0] = 0
    blit_min(canvas, glyph, 1, 1)
    assert int(canvas[1, 1]) == 0
    assert int(canvas[1, 2]) == 255
