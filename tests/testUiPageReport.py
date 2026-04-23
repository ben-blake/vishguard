"""Tests for pageReport pure helper functions (T4.1).

Streamlit render calls are not tested here — only the pure helper
functions that can be exercised without a running Streamlit context.
"""
from __future__ import annotations

import pytest

from vishguard.types import Tactic
from vishguard.ui.pageReport import (
    bandColor,
    formatPct,
    spoofLabel,
    tacticRows,
    timingsTable,
)


# ---------------------------------------------------------------------------
# bandColor
# ---------------------------------------------------------------------------

class TestBandColor:
    def test_returns_string_for_each_band(self):
        for band in ("low", "medium", "high", "critical"):
            result = bandColor(band)
            assert isinstance(result, str) and len(result) > 0

    def test_critical_differs_from_low(self):
        assert bandColor("critical") != bandColor("low")

    def test_unknown_band_returns_fallback_string(self):
        result = bandColor("unknown_band")
        assert isinstance(result, str) and len(result) > 0

    def test_all_four_bands_are_distinct(self):
        colors = [bandColor(b) for b in ("low", "medium", "high", "critical")]
        assert len(set(colors)) == 4


# ---------------------------------------------------------------------------
# spoofLabel
# ---------------------------------------------------------------------------

class TestSpoofLabel:
    def test_high_probability_is_synthetic(self):
        assert spoofLabel(0.9) == "Synthetic"

    def test_low_probability_is_real(self):
        assert spoofLabel(0.1) == "Real"

    def test_at_default_threshold_is_synthetic(self):
        assert spoofLabel(0.5) == "Synthetic"

    def test_just_below_threshold_is_real(self):
        assert spoofLabel(0.499) == "Real"

    def test_custom_threshold_respected(self):
        assert spoofLabel(0.4, threshold=0.4) == "Synthetic"
        assert spoofLabel(0.39, threshold=0.4) == "Real"


# ---------------------------------------------------------------------------
# formatPct
# ---------------------------------------------------------------------------

class TestFormatPct:
    def test_three_quarters(self):
        assert formatPct(0.75) == "75%"

    def test_zero(self):
        assert formatPct(0.0) == "0%"

    def test_one(self):
        assert formatPct(1.0) == "100%"

    def test_rounds_to_integer(self):
        result = formatPct(0.336)
        assert result.endswith("%")
        assert result[:-1].isdigit()


# ---------------------------------------------------------------------------
# tacticRows
# ---------------------------------------------------------------------------

class TestTacticRows:
    def test_single_tactic_returns_one_row(self):
        tactics = (Tactic(label="authority", confidence=0.9, evidenceSpans=("you owe money",)),)
        rows = tacticRows(tactics)
        assert len(rows) == 1

    def test_row_contains_label_and_confidence(self):
        tactics = (Tactic(label="urgency", confidence=0.7, evidenceSpans=()),)
        rows = tacticRows(tactics)
        assert rows[0]["label"] == "urgency"
        assert rows[0]["confidence"] == 0.7

    def test_empty_tactics_returns_empty_list(self):
        assert tacticRows(()) == []

    def test_multiple_tactics_preserves_order(self):
        tactics = (
            Tactic(label="pretexting", confidence=0.8, evidenceSpans=()),
            Tactic(label="benign", confidence=0.6, evidenceSpans=()),
        )
        rows = tacticRows(tactics)
        assert rows[0]["label"] == "pretexting"
        assert rows[1]["label"] == "benign"


# ---------------------------------------------------------------------------
# timingsTable
# ---------------------------------------------------------------------------

class TestTimingsTable:
    def test_nonempty_timings_return_rows(self):
        rows = timingsTable({"asr": 2.5, "ingestion": 0.1})
        assert len(rows) == 2

    def test_row_has_stage_and_duration_keys(self):
        rows = timingsTable({"asr": 2.5})
        assert "stage" in rows[0]
        assert "duration_s" in rows[0]

    def test_duration_value_matches_input(self):
        rows = timingsTable({"antiSpoof": 4.68})
        assert rows[0]["duration_s"] == 4.68

    def test_empty_dict_returns_empty_list(self):
        assert timingsTable({}) == []
