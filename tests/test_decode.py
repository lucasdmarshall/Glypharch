import numpy as np

from tinyocr.charset import CHARSET
from tinyocr.decode import greedy_decode


def _logits(indices: list[int], nclass: int) -> np.ndarray:
    arr = np.full((1, len(indices), nclass), -8.0, dtype=np.float32)
    for time, idx in enumerate(indices):
        arr[0, time, idx] = 0.0
    return arr


def test_greedy_collapses_repeats_and_keeps_blank_separated_repeats():
    assert greedy_decode(_logits([0, 1, 1, 0, 2], 3), "AB") == ["AB"]
    assert greedy_decode(_logits([1, 0, 1], 3), "AB") == ["AA"]


def test_greedy_batch():
    first = _logits([1, 0, 2, 0], 3)
    second = _logits([2, 2, 0, 1], 3)
    batch = np.concatenate([first, second], axis=0)
    assert greedy_decode(batch, "AB") == ["AB", "BA"]


def _one_hot(winners: list[tuple[str, str]]) -> np.ndarray:
    """Each step is (winner, runner-up). Winner scores 0, runner-up scores -1."""
    nclass = len(CHARSET) + 1
    arr = np.full((1, len(winners), nclass), -8.0, dtype=np.float32)
    for time, (first, second) in enumerate(winners):
        arr[0, time, CHARSET.index(first) + 1] = 0.0
        arr[0, time, CHARSET.index(second) + 1] = -1.0
    return arr


def test_digits_mask_forbids_o():
    logits = _one_hot([("O", "0")])
    assert greedy_decode(logits, CHARSET) == ["O"]
    assert greedy_decode(logits, CHARSET, alphabet="digits") == ["0"]


def test_letters_mask_forbids_zero_and_one():
    logits = _one_hot([("0", "O"), ("1", "I")])
    assert greedy_decode(logits, CHARSET) == ["01"]
    assert greedy_decode(logits, CHARSET, alphabet="letters") == ["OI"]
