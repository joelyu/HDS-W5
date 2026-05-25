"""Pure statistical-comparison helpers for paired classifier evaluation.

Two classifiers evaluated on the SAME test set give paired predictions, so the
right tools are McNemar's test (do their error rates differ?) and a bootstrap
confidence interval on a metric difference (how large is the gap, with what
uncertainty?). Both operate on per-image predictions, not on the multi-seed
mean +/- std (which only measures a model's stability to its own random seed,
not whether two models differ on the population).

No torch, no I/O — importable and unit-testable without data.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score
from statsmodels.stats.contingency_tables import mcnemar


def mcnemar_test(preds_a, preds_b, y_true) -> dict:
    """McNemar's test comparing two classifiers on the same test set.

    Discordance counts: b = A-correct & B-wrong, c = A-wrong & B-correct.
    Tests H0: equal error rates (b == c). Uses the chi-square statistic with
    continuity correction in the large-sample regime, and the exact binomial
    test when discordant pairs are few (b + c < 25).

    Returns {statistic, pvalue, b, c, n_discordant}.
    """
    preds_a = np.asarray(preds_a)
    preds_b = np.asarray(preds_b)
    y_true = np.asarray(y_true)
    if not (len(preds_a) == len(preds_b) == len(y_true)):
        raise ValueError("preds_a, preds_b, y_true must be the same length")

    correct_a = preds_a == y_true
    correct_b = preds_b == y_true
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n_disc = b + c

    if n_disc == 0:
        # Identical correctness on every sample — no evidence of a difference.
        return {"statistic": 0.0, "pvalue": 1.0, "b": b, "c": c, "n_discordant": 0}

    table = [
        [int(np.sum(correct_a & correct_b)), b],
        [c, int(np.sum(~correct_a & ~correct_b))],
    ]
    res = mcnemar(table, exact=(n_disc < 25), correction=True)
    return {
        "statistic": float(res.statistic),
        "pvalue": float(res.pvalue),
        "b": b,
        "c": c,
        "n_discordant": n_disc,
    }


def _metric_fn(metric: str):
    if metric == "accuracy":
        return lambda y, p: float(np.mean(y == p))
    avg = {"macro_f1": "macro", "weighted_f1": "weighted"}.get(metric)
    if avg is None:
        raise ValueError(f"unknown metric {metric!r}")
    return lambda y, p: float(f1_score(y, p, average=avg, zero_division=0))


def bootstrap_delta(
    preds_a, preds_b, y_true, metric: str = "macro_f1", n_boot: int = 2000, seed: int = 42
) -> dict:
    """Bootstrap 95% CI on metric(A) - metric(B) over the test set.

    Resamples test indices with replacement n_boot times, recomputing the
    metric difference each time. Returns the point estimate (full sample) plus
    the 2.5/97.5 percentile CI of the bootstrap distribution. A CI excluding 0
    is the effect-size complement to McNemar's p-value.

    metric: 'macro_f1' (default), 'weighted_f1', or 'accuracy'. macro_f1 uses
    zero_division=0 so a rare class absent from a resample scores 0 rather than
    erroring.

    Returns {delta, ci_low, ci_high, metric, n_boot}.
    """
    preds_a = np.asarray(preds_a)
    preds_b = np.asarray(preds_b)
    y_true = np.asarray(y_true)
    if not (len(preds_a) == len(preds_b) == len(y_true)):
        raise ValueError("preds_a, preds_b, y_true must be the same length")

    score = _metric_fn(metric)
    point = score(y_true, preds_a) - score(y_true, preds_b)

    n = len(y_true)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[i] = score(y_true[idx], preds_a[idx]) - score(y_true[idx], preds_b[idx])

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "delta": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "metric": metric,
        "n_boot": n_boot,
    }


def holm_correction(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, returned in input order.

    Controls the family-wise error rate across the comparison set without the
    conservatism of plain Bonferroni. Adjusted values are made monotone
    non-decreasing along the sorted order and clipped to 1.0.
    """
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)  # enforce monotonicity along sorted order
        adjusted[idx] = min(running, 1.0)
    return [float(x) for x in adjusted]
