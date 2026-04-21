"""Unit tests for eval/runLatencyBench.py helper functions (T3.6)."""
import pytest

from eval.runLatencyBench import summarizeTimings, makeTimingRow


# ---------------------------------------------------------------------------
# summarizeTimings — aggregate per-stage latency across runs
# ---------------------------------------------------------------------------

_RUNS = [
    {"load": 0.1, "asr": 1.0, "spoof": 0.5, "tactic": 2.0, "risk": 0.1, "tts": 0.3, "total": 4.0},
    {"load": 0.2, "asr": 1.2, "spoof": 0.6, "tactic": 2.2, "risk": 0.2, "tts": 0.4, "total": 4.8},
]


def test_summarize_timings_mean_keys():
    summary = summarizeTimings(_RUNS)
    for key in ("load", "asr", "spoof", "tactic", "risk", "tts", "total"):
        assert f"mean_{key}_s" in summary


def test_summarize_timings_mean_values():
    summary = summarizeTimings(_RUNS)
    assert summary["mean_asr_s"] == pytest.approx(1.1)
    assert summary["mean_total_s"] == pytest.approx(4.4)


def test_summarize_timings_all_positive():
    summary = summarizeTimings(_RUNS)
    assert all(v >= 0 for v in summary.values())


def test_summarize_single_run():
    single = [{"load": 0.5, "asr": 2.0, "spoof": 0.8, "tactic": 3.0, "risk": 0.2, "tts": 0.5, "total": 7.0}]
    summary = summarizeTimings(single)
    assert summary["mean_total_s"] == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# makeTimingRow — build one CSV row from orchestrator timings dict
# ---------------------------------------------------------------------------

def test_make_timing_row_includes_clip_info():
    row = makeTimingRow(clip_path="test.wav", duration_sec=15.0, timings={"load": 0.1, "asr": 1.0})
    assert row["clip_path"] == "test.wav"
    assert row["duration_sec"] == pytest.approx(15.0)


def test_make_timing_row_has_all_stage_keys():
    timings = {"load": 0.1, "asr": 1.0, "spoof": 0.5, "tactic": 2.0, "risk": 0.1, "tts": 0.3, "total": 4.0}
    row = makeTimingRow("x.wav", 45.0, timings)
    for k, v in timings.items():
        assert row[k] == pytest.approx(v)


def test_make_timing_row_total_field():
    timings = {"asr": 1.0, "total": 5.0}
    row = makeTimingRow("x.wav", 90.0, timings)
    assert row["total"] == pytest.approx(5.0)
