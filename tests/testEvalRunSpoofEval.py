"""Unit tests for eval/runSpoofEval.py helper functions (T3.2)."""
import pytest

from eval.runSpoofEval import computeEer, computeMetrics


# ---------------------------------------------------------------------------
# computeMetrics — accuracy + F1 on binary preds
# ---------------------------------------------------------------------------

def test_compute_metrics_perfect():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    m = computeMetrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)


def test_compute_metrics_all_wrong():
    y_true = [0, 0, 1, 1]
    y_pred = [1, 1, 0, 0]
    m = computeMetrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(0.0)
    assert m["f1"] == pytest.approx(0.0)


def test_compute_metrics_mixed():
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 0, 0]
    m = computeMetrics(y_true, y_pred)
    assert 0 < m["accuracy"] < 1
    assert 0 < m["f1"] < 1


def test_compute_metrics_returns_required_keys():
    m = computeMetrics([0, 1], [0, 1])
    assert {"accuracy", "f1", "precision", "recall"} <= set(m.keys())


# ---------------------------------------------------------------------------
# computeEer — equal error rate
# ---------------------------------------------------------------------------

def test_eer_perfect_separation():
    # real=0, fake=1; scores perfectly ordered
    y_true = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    eer = computeEer(y_true, scores)
    assert eer == pytest.approx(0.0, abs=0.05)


def test_eer_random_chance():
    # All scores equal → EER ≈ 0.5
    y_true = [0, 0, 1, 1]
    scores = [0.5, 0.5, 0.5, 0.5]
    eer = computeEer(y_true, scores)
    assert eer == pytest.approx(0.5, abs=0.1)


def test_eer_between_zero_and_one():
    y_true = [0, 1, 0, 1, 0, 1]
    scores = [0.3, 0.7, 0.4, 0.6, 0.55, 0.45]
    eer = computeEer(y_true, scores)
    assert 0.0 <= eer <= 1.0
