import numpy as np
import pytest

from stats import bootstrap_delta, holm_correction, mcnemar_test


# ── McNemar ──────────────────────────────────────────────────────────────────


def test_mcnemar_identical_predictions_not_significant():
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    preds = y.copy()
    out = mcnemar_test(preds, preds, y)
    assert out["b"] == 0 and out["c"] == 0
    assert out["n_discordant"] == 0
    assert out["pvalue"] == 1.0


def test_mcnemar_a_dominates_is_significant():
    # A correct everywhere, B wrong everywhere -> all discordant pairs favour A.
    n = 100
    y = np.zeros(n, dtype=int)
    preds_a = y.copy()
    preds_b = np.ones(n, dtype=int)
    out = mcnemar_test(preds_a, preds_b, y)
    assert out["b"] == n and out["c"] == 0
    assert out["pvalue"] < 0.05


def test_mcnemar_returns_expected_keys():
    y = np.array([0, 1, 0, 1, 0, 1])
    a = np.array([0, 1, 0, 0, 0, 1])  # one extra error vs y at idx 3
    b = np.array([0, 0, 0, 1, 0, 1])  # one extra error vs y at idx 1
    out = mcnemar_test(a, b, y)
    assert set(out) == {"statistic", "pvalue", "b", "c", "n_discordant"}
    assert out["b"] == 1 and out["c"] == 1  # symmetric discordance


def test_mcnemar_length_mismatch_raises():
    with pytest.raises(ValueError):
        mcnemar_test([0, 1], [0], [0, 1])


# ── Bootstrap ────────────────────────────────────────────────────────────────


def test_bootstrap_identical_predictions_ci_contains_zero():
    y = np.array([0, 1, 2] * 20)
    preds = y.copy()
    out = bootstrap_delta(preds, preds, y, metric="macro_f1", n_boot=200)
    assert out["delta"] == 0.0
    assert out["ci_low"] <= 0.0 <= out["ci_high"]


def test_bootstrap_clear_winner_ci_excludes_zero():
    # A perfect, B chance-level on accuracy -> delta ~1, CI well above 0.
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 300)
    preds_a = y.copy()
    preds_b = rng.integers(0, 3, 300)
    out = bootstrap_delta(preds_a, preds_b, y, metric="accuracy", n_boot=500)
    assert out["delta"] > 0.5
    assert out["ci_low"] > 0.0


def test_bootstrap_is_deterministic_with_seed():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 4, 200)
    a = rng.integers(0, 4, 200)
    b = rng.integers(0, 4, 200)
    out1 = bootstrap_delta(a, b, y, n_boot=300, seed=7)
    out2 = bootstrap_delta(a, b, y, n_boot=300, seed=7)
    assert out1 == out2


def test_bootstrap_unknown_metric_raises():
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError):
        bootstrap_delta(y, y, y, metric="not_a_metric", n_boot=10)


# ── Holm correction ──────────────────────────────────────────────────────────


def test_holm_preserves_order_and_inflates():
    raw = [0.01, 0.04, 0.03]
    adj = holm_correction(raw)
    assert len(adj) == 3
    # each adjusted >= raw, all <= 1
    assert all(a >= r - 1e-12 for a, r in zip(adj, raw))
    assert all(a <= 1.0 for a in adj)
    # smallest raw p gets multiplied by m=3
    assert abs(adj[0] - 0.03) < 1e-9


def test_holm_empty():
    assert holm_correction([]) == []


def test_holm_is_monotone_along_sorted_order():
    raw = [0.001, 0.5, 0.04, 0.2]
    adj = holm_correction(raw)
    # sort pairs by raw p, adjusted must be non-decreasing
    pairs = sorted(zip(raw, adj))
    sorted_adj = [a for _, a in pairs]
    assert sorted_adj == sorted(sorted_adj)
