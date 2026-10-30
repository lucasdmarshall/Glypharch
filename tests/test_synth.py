import numpy as np
import pytest

from tinyocr.charset import CHARSET
from tinyocr.fonts import TRAIN_FILENAMES, resolve_fonts
from tinyocr.synth import STANDARD, SynthSource, child_seed


def test_child_seed_is_stable_and_changes_with_index():
    assert child_seed(10, 1) == child_seed(10, 1)
    assert child_seed(10, 1) != child_seed(10, 2)


def test_sample_is_deterministic_and_in_charset():
    fonts = resolve_fonts(TRAIN_FILENAMES, None)
    if len(fonts) < 1:
        pytest.skip("no fonts installed")
    source = SynthSource(fonts[:2], sizes=(22,))
    first, text = source.sample(3, seed=7, augment=STANDARD)
    again, text_again = source.sample(3, seed=7, augment=STANDARD)
    assert text == text_again
    assert np.array_equal(first, again)
    assert 4 <= len(text) <= 8
    assert set(text) <= set(CHARSET)
    assert first.shape == (1, 32, 160)
    tall, _ = source.sample(3, seed=7, augment=STANDARD, height=64, width=160)
    assert tall.shape == (1, 64, 160)
