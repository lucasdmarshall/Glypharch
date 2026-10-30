"""Greedy CTC decode. One pass, no language model, no beam search."""

import numpy as np

from tinyocr.config import BLANK


def greedy_decode(log_probs: np.ndarray, charset: str) -> list[str]:
    """Decode a batch of log-probabilities shaped (N, T, C)."""
    if log_probs.ndim != 3:
        raise ValueError(f"expected (N, T, C), got {log_probs.shape}")
    if log_probs.shape[-1] != len(charset) + 1:
        raise ValueError(
            f"class count {log_probs.shape[-1]} does not match charset "
            f"length {len(charset)} plus blank"
        )
    indices = np.argmax(log_probs, axis=-1)
    texts: list[str] = []
    for seq in indices:
        previous = BLANK
        chars: list[str] = []
        for raw in seq.tolist():
            idx = int(raw)
            if idx != BLANK and idx != previous:
                chars.append(charset[idx - 1])
            previous = idx
        texts.append("".join(chars))
    return texts
