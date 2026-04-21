"""T2.7: Tests for reportSchema — JSON round-trip and file validation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from vishguard.reportSchema import fromDict, saveReport, toDict, validateReport
from vishguard.types import (
    AudioMeta,
    RiskReport,
    RiskScore,
    SpoofVerdict,
    Tactic,
    Transcript,
    TranscriptSegment,
)


def _sample_report(briefing: bool = False) -> RiskReport:
    return RiskReport(
        version="1.0.0",
        audio=AudioMeta(
            sourcePath=Path("samples/call_01.wav"),
            durationSec=47.2,
            sampleRate=16_000,
        ),
        transcript=Transcript(
            fullText="Hello, this is calling from the IRS.",
            segments=(
                TranscriptSegment(
                    startSec=0.0,
                    endSec=4.1,
                    text="Hello, this is calling from the IRS.",
                ),
            ),
            languageCode="en",
            modelId="openai/whisper-small",
        ),
        spoof=SpoofVerdict(
            pSynthetic=0.87,
            modelId="mo-thecreator/Deepfake-audio-detection",
            rationale="High-confidence synthetic.",
        ),
        tactics=(
            Tactic(
                label="authority",
                confidence=0.93,
                evidenceSpans=("this is calling from the IRS",),
            ),
        ),
        risk=RiskScore(score=85, band="critical", reasoning="High risk. Block and report."),
        briefingAudioPath=Path("out/briefing.wav") if briefing else None,
        timings={"ingestion": 0.1, "asr": 8.4, "total": 8.5},
    )


class TestToDict:
    def test_returns_dict(self) -> None:
        assert isinstance(toDict(_sample_report()), dict)

    def test_required_top_level_keys(self) -> None:
        d = toDict(_sample_report())
        for key in ("version", "audio", "transcript", "spoof", "tactics", "risk", "timings"):
            assert key in d, f"Missing key: {key}"

    def test_source_path_is_string(self) -> None:
        d = toDict(_sample_report())
        assert isinstance(d["audio"]["sourcePath"], str)

    def test_briefing_path_none_when_unset(self) -> None:
        d = toDict(_sample_report(briefing=False))
        assert d["briefingAudioPath"] is None

    def test_briefing_path_is_string_when_set(self) -> None:
        d = toDict(_sample_report(briefing=True))
        assert isinstance(d["briefingAudioPath"], str)

    def test_tactics_is_list(self) -> None:
        d = toDict(_sample_report())
        assert isinstance(d["tactics"], list)


class TestFromDict:
    def test_roundtrip_version(self) -> None:
        original = _sample_report()
        restored = fromDict(toDict(original))
        assert restored.version == original.version

    def test_roundtrip_risk_score(self) -> None:
        original = _sample_report()
        restored = fromDict(toDict(original))
        assert restored.risk.score == original.risk.score
        assert restored.risk.band == original.risk.band

    def test_roundtrip_spoof(self) -> None:
        original = _sample_report()
        restored = fromDict(toDict(original))
        assert restored.spoof.pSynthetic == original.spoof.pSynthetic

    def test_roundtrip_tactics_count(self) -> None:
        original = _sample_report()
        restored = fromDict(toDict(original))
        assert len(restored.tactics) == len(original.tactics)

    def test_roundtrip_tactic_label(self) -> None:
        original = _sample_report()
        restored = fromDict(toDict(original))
        assert restored.tactics[0].label == original.tactics[0].label

    def test_briefing_path_none(self) -> None:
        restored = fromDict(toDict(_sample_report(briefing=False)))
        assert restored.briefingAudioPath is None

    def test_briefing_path_restored_as_path(self) -> None:
        restored = fromDict(toDict(_sample_report(briefing=True)))
        assert isinstance(restored.briefingAudioPath, Path)

    def test_segments_preserved(self) -> None:
        original = _sample_report()
        restored = fromDict(toDict(original))
        assert len(restored.transcript.segments) == 1
        assert restored.transcript.segments[0].startSec == 0.0


class TestSaveAndValidate:
    def test_save_creates_json_file(self, tmp_path: Path) -> None:
        out = tmp_path / "report.json"
        saveReport(_sample_report(), out)
        assert out.exists()

    def test_save_writes_valid_json(self, tmp_path: Path) -> None:
        out = tmp_path / "report.json"
        saveReport(_sample_report(), out)
        with open(out) as f:
            data = json.load(f)
        assert data["version"] == "1.0.0"

    def test_validate_returns_risk_report(self, tmp_path: Path) -> None:
        out = tmp_path / "report.json"
        saveReport(_sample_report(), out)
        loaded = validateReport(out)
        assert isinstance(loaded, RiskReport)

    def test_validate_preserves_risk_score(self, tmp_path: Path) -> None:
        out = tmp_path / "report.json"
        saveReport(_sample_report(), out)
        loaded = validateReport(out)
        assert loaded.risk.score == 85

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        out = tmp_path / "sub" / "dir" / "report.json"
        saveReport(_sample_report(), out)
        assert out.exists()
