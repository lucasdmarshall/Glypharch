from tinyocr.metrics import edit_distance, score_predictions


def test_edit_distance_substitution():
    dist, subs = edit_distance("B0", "A0")
    assert dist == 1
    assert subs == [("A", "B")]


def test_edit_distance_insertion():
    dist, subs = edit_distance("A", "AA")
    assert dist == 1
    assert subs == []


def test_score_exact_and_character_accuracy():
    scored = score_predictions(["AB", "A0"], ["AB", "AA"])
    assert scored["n"] == 2
    assert scored["exact_accuracy"] == 0.5
    assert scored["character_error_rate"] == 1 / 4
    assert scored["top_substitutions"] == [{"change": "A->0", "count": 1}]
