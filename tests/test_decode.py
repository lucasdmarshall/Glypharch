import numpy as np

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
