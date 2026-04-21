"""T2.5: Tests for riskSynthesis — pure scoring formula and band assignment."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from vishguard.riskSynthesis import _assignBand, _computeScore
from vishguard.types import (
    LlmConfig,
    RiskScore,
    SpoofVerdict,
    Tactic,
    Transcript,
    TranscriptSegment,
)


def _spoof(p: float) -> SpoofVerdict:
    return SpoofVerdict(pSynthetic=p, modelId="test", rationale="test")


def _tactic(label: str, conf: float) -> Tactic:
    return Tactic(label=label, confidence=conf, evidenceSpans=())


def _transcript(text: str = "test") -> Transcript:
    seg = TranscriptSegment(startSec=0.0, endSec=1.0, text=text)
    return Transcript(fullText=text, segments=(seg,), languageCode="en", modelId="test")


class TestComputeScore:
    def test_low_scenario(self) -> None:
        # spoof=0.1 → 4pts; benign penalty → -20; clamped to 0
        score = _computeScore(_spoof(0.1), (_tactic("benign", 0.99),))
        assert score < 25, f"Expected low (<25), got {score}"

    def test_medium_scenario(self) -> None:
        # spoof=0.5 → 20pts; urgency+pretexting → +18pts; total=38
        score = _computeScore(
            _spoof(0.5),
            (_tactic("urgency", 0.7), _tactic("pretexting", 0.5)),
        )
        assert 25 <= score < 50, f"Expected medium (25-49), got {score}"

    def test_critical_scenario(self) -> None:
        # spoof=0.9 → 36pts; 3 malicious tactics → +40.5pts capped at 60; total≥75
        score = _computeScore(
            _spoof(0.9),
            (
                _tactic("authority", 0.95),
                _tactic("fear_intimidation", 0.90),
                _tactic("financial_manipulation", 0.85),
            ),
        )
        assert score >= 75, f"Expected critical (>=75), got {score}"

    def test_score_clamped_to_100(self) -> None:
        score = _computeScore(_spoof(1.0), tuple(_tactic("authority", 1.0) for _ in range(10)))
        assert score <= 100

    def test_score_non_negative(self) -> None:
        score = _computeScore(_spoof(0.0), ())
        assert score >= 0

    def test_benign_label_subtracts_from_score(self) -> None:
        # spoof=0.3 without benign → 12; with benign → max(0, 12-20)=0
        without_benign = _computeScore(_spoof(0.3), ())
        with_benign = _computeScore(_spoof(0.3), (_tactic("benign", 0.99),))
        assert with_benign <= without_benign

    def test_benign_not_counted_as_malicious_tactic(self) -> None:
        # benign should not add to malicious confidence sum
        score_no_tactics = _computeScore(_spoof(0.5), ())
        score_with_benign = _computeScore(_spoof(0.5), (_tactic("benign", 1.0),))
        # benign only reduces, never increases above the spoof component
        assert score_with_benign <= score_no_tactics


class TestAssignBand:
    def test_critical_band_at_75(self) -> None:
        assert _assignBand(75) == "critical"

    def test_critical_band_at_100(self) -> None:
        assert _assignBand(100) == "critical"

    def test_high_band_at_50(self) -> None:
        assert _assignBand(50) == "high"

    def test_high_band_at_74(self) -> None:
        assert _assignBand(74) == "high"

    def test_medium_band_at_25(self) -> None:
        assert _assignBand(25) == "medium"

    def test_medium_band_at_49(self) -> None:
        assert _assignBand(49) == "medium"

    def test_low_band_at_0(self) -> None:
        assert _assignBand(0) == "low"

    def test_low_band_at_24(self) -> None:
        assert _assignBand(24) == "low"


class TestSynthesizeRisk:
    def test_returns_risk_score(self) -> None:
        from vishguard.riskSynthesis import synthesizeRisk

        cfg = LlmConfig(modelId="test-llm", device="cpu", loadIn4Bit=False)

        with patch("vishguard.riskSynthesis._callLlmForReasoning", return_value="High risk call."):
            result = synthesizeRisk(
                _transcript(),
                _spoof(0.9),
                (_tactic("authority", 0.95),),
                cfg,
            )
        assert isinstance(result, RiskScore)

    def test_reasoning_from_llm(self) -> None:
        from vishguard.riskSynthesis import synthesizeRisk

        cfg = LlmConfig(modelId="test-llm", device="cpu", loadIn4Bit=False)

        with patch("vishguard.riskSynthesis._callLlmForReasoning", return_value="Suspicious call."):
            result = synthesizeRisk(_transcript(), _spoof(0.5), (), cfg)
        assert result.reasoning == "Suspicious call."

    def test_band_consistent_with_score(self) -> None:
        from vishguard.riskSynthesis import synthesizeRisk

        cfg = LlmConfig(modelId="test-llm", device="cpu", loadIn4Bit=False)

        with patch("vishguard.riskSynthesis._callLlmForReasoning", return_value="ok"):
            result = synthesizeRisk(
                _transcript(),
                _spoof(0.9),
                (_tactic("authority", 0.95), _tactic("fear_intimidation", 0.9)),
                cfg,
            )
        assert result.band == _assignBand(result.score)
