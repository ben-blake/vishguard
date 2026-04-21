"""Unit tests for eval/runAsrEval.py helper functions (T3.5)."""
import numpy as np
import pytest

from eval.runAsrEval import normalizeText, addNoise, computeWer


# ---------------------------------------------------------------------------
# normalizeText — strip punctuation, lowercase
# ---------------------------------------------------------------------------

def test_normalize_strips_punctuation():
    assert normalizeText("Hello, World!") == "hello world"


def test_normalize_lowercases():
    assert normalizeText("UPPERCASE TEXT") == "uppercase text"


def test_normalize_removes_double_spaces():
    result = normalizeText("word.  another")
    assert "  " not in result


def test_normalize_empty_string():
    assert normalizeText("") == ""


def test_normalize_numbers_preserved():
    result = normalizeText("There are 3 items.")
    assert "3" in result


# ---------------------------------------------------------------------------
# addNoise — add gaussian noise at given SNR
# ---------------------------------------------------------------------------

def test_add_noise_output_shape():
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(16000).astype(np.float32)
    noisy = addNoise(signal, snr_db=20.0)
    assert noisy.shape == signal.shape


def test_add_noise_changes_signal():
    rng = np.random.default_rng(1)
    signal = rng.standard_normal(16000).astype(np.float32)
    noisy = addNoise(signal, snr_db=10.0)
    assert not np.allclose(signal, noisy)


def test_add_noise_higher_snr_closer_to_original():
    rng = np.random.default_rng(2)
    signal = rng.standard_normal(16000).astype(np.float32)
    noisy_high = addNoise(signal, snr_db=30.0)
    noisy_low = addNoise(signal, snr_db=5.0)
    diff_high = float(np.mean(np.abs(noisy_high - signal)))
    diff_low = float(np.mean(np.abs(noisy_low - signal)))
    assert diff_high < diff_low


# ---------------------------------------------------------------------------
# computeWer — word error rate
# ---------------------------------------------------------------------------

def test_wer_perfect():
    assert computeWer("hello world", "hello world") == pytest.approx(0.0)


def test_wer_one_substitution():
    wer = computeWer("hello world", "hello earth")
    assert 0 < wer <= 1.0


def test_wer_completely_wrong():
    wer = computeWer("hello world", "foo bar baz")
    assert wer > 0


def test_wer_empty_reference_returns_zero():
    # empty reference → undefined; convention: 0.0
    assert computeWer("", "") == pytest.approx(0.0)
