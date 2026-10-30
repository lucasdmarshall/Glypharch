"""Exact-line accuracy and character accuracy from edit distance."""

from collections import Counter


def edit_distance(pred: str, truth: str) -> tuple[int, list[tuple[str, str]]]:
    """Levenshtein distance and the substituted (truth, pred) pairs."""
    n, m = len(pred), len(truth)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        pi = pred[i - 1]
        for j in range(1, m + 1):
            cost = pi != truth[j - 1]
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    subs: list[tuple[str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = int(pred[i - 1] != truth[j - 1])
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                if cost:
                    subs.append((truth[j - 1], pred[i - 1]))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            i -= 1
            continue
        j -= 1
    subs.reverse()
    return dp[n][m], subs


def score_predictions(preds: list[str], truths: list[str]) -> dict:
    if len(preds) != len(truths) or not truths:
        raise ValueError("predictions and labels must be the same non-empty length")
    exact = 0
    dist = 0
    chars = 0
    subs: Counter[str] = Counter()
    for pred, truth in zip(preds, truths):
        if pred == truth:
            exact += 1
        d, pairs = edit_distance(pred, truth)
        dist += d
        chars += len(truth)
        subs.update(f"{src}->{dst}" for src, dst in pairs)
    cer = dist / chars if chars else 0.0
    return {
        "n": len(truths),
        "exact_accuracy": exact / len(truths),
        "character_accuracy": 1.0 - cer,
        "character_error_rate": cer,
        "top_substitutions": [
            {"change": name, "count": count} for name, count in subs.most_common(8)
        ],
    }
