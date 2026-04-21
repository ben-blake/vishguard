"""Unit tests for eval/runTacticEval.py helper functions (T3.4)."""
import pytest

from eval.runTacticEval import computePerLabelF1, computeMacroF1
from vishguard.types import TACTIC_TAXONOMY


# ---------------------------------------------------------------------------
# computePerLabelF1
# ---------------------------------------------------------------------------

def test_per_label_f1_perfect():
    label = "urgency"
    y_true = [{"urgency"}, {"authority"}, {"urgency"}]
    y_pred = [{"urgency"}, {"authority"}, {"urgency"}]
    assert computePerLabelF1(label, y_true, y_pred) == pytest.approx(1.0)


def test_per_label_f1_zero():
    label = "urgency"
    y_true = [{"urgency"}, {"urgency"}]
    y_pred = [{"authority"}, {"authority"}]
    assert computePerLabelF1(label, y_true, y_pred) == pytest.approx(0.0)


def test_per_label_f1_partial():
    label = "urgency"
    y_true = [{"urgency"}, {"urgency"}, {"authority"}]
    y_pred = [{"urgency"}, {"authority"}, {"authority"}]
    f1 = computePerLabelF1(label, y_true, y_pred)
    assert 0.0 < f1 < 1.0


def test_per_label_f1_label_absent_returns_zero():
    label = "tech_support"
    y_true = [{"urgency"}, {"authority"}]
    y_pred = [{"urgency"}, {"authority"}]
    # label never appears → F1 defined as 0
    f1 = computePerLabelF1(label, y_true, y_pred)
    assert f1 == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# computeMacroF1
# ---------------------------------------------------------------------------

def test_macro_f1_perfect():
    y_true = [{"urgency"}, {"authority"}]
    y_pred = [{"urgency"}, {"authority"}]
    labels = ("urgency", "authority")
    assert computeMacroF1(y_true, y_pred, labels) == pytest.approx(1.0)


def test_macro_f1_between_zero_and_one():
    y_true = [{"urgency"}, {"urgency"}, {"authority"}]
    y_pred = [{"urgency"}, {"authority"}, {"authority"}]
    labels = ("urgency", "authority")
    f1 = computeMacroF1(y_true, y_pred, labels)
    assert 0.0 <= f1 <= 1.0


def test_macro_f1_uses_all_labels():
    y_true = [{"urgency"}]
    y_pred = [{"urgency"}]
    labels = TACTIC_TAXONOMY  # 10 labels; most score 0
    f1 = computeMacroF1(y_true, y_pred, labels)
    # macro average depressed by all the missing labels
    assert f1 < 0.5
