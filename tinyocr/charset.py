"""Label encoding. Index 0 is the CTC blank, so digits are not blank."""

from tinyocr.config import BLANK, CHARSET


def encode(text: str) -> list[int]:
    ids: list[int] = []
    for ch in text:
        pos = CHARSET.find(ch)
        if pos < 0:
            raise ValueError(
                f"character {ch!r} is outside the charset {CHARSET!r}"
            )
        ids.append(pos + 1)
    if any(i == BLANK for i in ids):
        raise RuntimeError("encoder produced a blank index")
    return ids


def normalize_label(text: str) -> str:
    cleaned = text.strip().upper()
    if not cleaned:
        raise ValueError("empty label")
    encode(cleaned)
    return cleaned
