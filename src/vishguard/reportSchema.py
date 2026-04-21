"""JSON schema validation and serialization for report.json artifacts.

Single source of truth for the JSON contract documented in
docs/ARCHITECTURE.md §4. Every eval script must validate through here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from .types import (
    AudioMeta,
    RiskReport,
    RiskScore,
    SpoofVerdict,
    Tactic,
    Transcript,
    TranscriptSegment,
)


class _SegmentModel(BaseModel):
    startSec: float
    endSec: float
    text: str


class _TranscriptModel(BaseModel):
    fullText: str
    segments: list[_SegmentModel]
    languageCode: str
    modelId: str


class _AudioMetaModel(BaseModel):
    sourcePath: str
    durationSec: float
    sampleRate: int


class _SpoofVerdictModel(BaseModel):
    pSynthetic: float
    modelId: str
    rationale: str


class _TacticModel(BaseModel):
    label: str
    confidence: float
    evidenceSpans: list[str]


class _RiskScoreModel(BaseModel):
    score: int
    band: str
    reasoning: str


class _RiskReportModel(BaseModel):
    version: str
    audio: _AudioMetaModel
    transcript: _TranscriptModel
    spoof: _SpoofVerdictModel
    tactics: list[_TacticModel]
    risk: _RiskScoreModel
    briefingAudioPath: Optional[str] = None
    timings: dict[str, float]


def toDict(report: RiskReport) -> dict:
    return {
        "version": report.version,
        "audio": {
            "sourcePath": str(report.audio.sourcePath),
            "durationSec": report.audio.durationSec,
            "sampleRate": report.audio.sampleRate,
        },
        "transcript": {
            "fullText": report.transcript.fullText,
            "segments": [
                {"startSec": s.startSec, "endSec": s.endSec, "text": s.text}
                for s in report.transcript.segments
            ],
            "languageCode": report.transcript.languageCode,
            "modelId": report.transcript.modelId,
        },
        "spoof": {
            "pSynthetic": report.spoof.pSynthetic,
            "modelId": report.spoof.modelId,
            "rationale": report.spoof.rationale,
        },
        "tactics": [
            {
                "label": t.label,
                "confidence": t.confidence,
                "evidenceSpans": list(t.evidenceSpans),
            }
            for t in report.tactics
        ],
        "risk": {
            "score": report.risk.score,
            "band": report.risk.band,
            "reasoning": report.risk.reasoning,
        },
        "briefingAudioPath": (
            str(report.briefingAudioPath) if report.briefingAudioPath else None
        ),
        "timings": report.timings,
    }


def fromDict(data: dict) -> RiskReport:
    m = _RiskReportModel.model_validate(data)
    return RiskReport(
        version=m.version,
        audio=AudioMeta(
            sourcePath=Path(m.audio.sourcePath),
            durationSec=m.audio.durationSec,
            sampleRate=m.audio.sampleRate,
        ),
        transcript=Transcript(
            fullText=m.transcript.fullText,
            segments=tuple(
                TranscriptSegment(startSec=s.startSec, endSec=s.endSec, text=s.text)
                for s in m.transcript.segments
            ),
            languageCode=m.transcript.languageCode,
            modelId=m.transcript.modelId,
        ),
        spoof=SpoofVerdict(
            pSynthetic=m.spoof.pSynthetic,
            modelId=m.spoof.modelId,
            rationale=m.spoof.rationale,
        ),
        tactics=tuple(
            Tactic(
                label=t.label,
                confidence=t.confidence,
                evidenceSpans=tuple(t.evidenceSpans),
            )
            for t in m.tactics
        ),
        risk=RiskScore(
            score=m.risk.score,
            band=m.risk.band,
            reasoning=m.risk.reasoning,
        ),
        briefingAudioPath=Path(m.briefingAudioPath) if m.briefingAudioPath else None,
        timings=m.timings,
    )


def saveReport(report: RiskReport, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(toDict(report), f, indent=2)


def validateReport(reportPath: Path) -> RiskReport:
    with open(reportPath) as f:
        data = json.load(f)
    return fromDict(data)
