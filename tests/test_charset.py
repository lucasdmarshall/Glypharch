import pytest

from tinyocr.charset import encode, normalize_label
from tinyocr.config import CHARSET


def test_digit_zero_is_not_the_ctc_blank():
    assert encode("0") == [1]
    assert encode("A") == [CHARSET.index("A") + 1]


def test_roundtrip_indexes_stay_inside_alphabet():
    ids = encode(CHARSET)
    assert ids[0] == 1
    assert ids[-1] == len(CHARSET)
    assert 0 not in ids


def test_unknown_character_is_rejected():
    with pytest.raises(ValueError):
        encode("A B")
    with pytest.raises(ValueError):
        encode("က")


def test_normalize_uppercases():
    assert normalize_label(" ab12 ") == "AB12"
