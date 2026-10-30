"""Greedy CTC decode. One pass, no language model, no beam search."""

import numpy as np

from tinyocr.config import BLANK

# Named alphabets for a field whose format is already known.
# Digits forbids O (and every other letter). Letters forbids 0 and 1
# (and every other digit). Random codes do not use this mask.
NAMED_ALPHABETS = {
    "digits": "0123456789",
    "letters": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
}


def mask_alphabet(log_probs: np.ndarray, charset: str, alphabet: str) -> np.ndarray:
    """Push disallowed non-blank classes below every real log-probability."""
    resolved = NAMED_ALPHABETS.get(alphabet, alphabet)
    allowed = set(resolved)
    unknown = [char for char in resolved if char not in charset]
    if unknown:
        raise ValueError(f"alphabet is outside the charset: {''.join(unknown)}")
    banned = [index + 1 for index, char in enumerate(charset) if char not in allowed]
    if not banned:
        return log_probs
    masked = np.array(log_probs, dtype=np.float32, copy=True)
    masked[..., banned] = np.float32(-1.0e4)
    return masked


def greedy_decode(
    log_probs: np.ndarray, charset: str, alphabet: str | None = None
) -> list[str]:
    """Decode a batch of log-probabilities shaped (N, T, C)."""
    if log_probs.ndim != 3:
        raise ValueError(f"expected (N, T, C), got {log_probs.shape}")
    if log_probs.shape[-1] != len(charset) + 1:
        raise ValueError(
            f"class count {log_probs.shape[-1]} does not match charset "
            f"length {len(charset)} plus blank"
        )
    if alphabet is not None:
        log_probs = mask_alphabet(log_probs, charset, alphabet)
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
