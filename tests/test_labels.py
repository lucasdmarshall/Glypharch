from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tinyocr.train import read_labels


def test_reads_tab_separated_labels(tmp_path: Path):
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    image_a.write_bytes(b"not-decoded-here")
    image_b.write_bytes(b"not-decoded-here")
    labels = tmp_path / "labels.tsv"
    labels.write_text("# comment\n\na.png\tab12\nb.png\tZ9\n", encoding="utf-8")
    rows = read_labels(labels)
    assert rows == [(image_a, "AB12"), (image_b, "Z9")]


def test_rejects_a_missing_column(tmp_path: Path):
    labels = tmp_path / "labels.tsv"
    labels.write_text("only-one-field\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        read_labels(labels)
