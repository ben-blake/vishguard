"""Pipeline orchestrator — wires every stage, captures per-stage latency."""
from __future__ import annotations

import time
from pathlib import Path

from .antiSpoof import detectSpoof
from .asrWhisper import transcribe
from .briefingTts import narrateBriefing
from .loadAudio import ingest
from .riskSynthesis import synthesizeRisk
from .reportSchema import saveReport
from .tacticClassifier import classifyTactics
from .types import (
    AsrConfig,
    AudioMeta,
    LlmConfig,
    RiskReport,
    SpoofConfig,
    TtsConfig,
)


def runPipeline(
    audioPath: Path,
    outDir: Path,
    asrCfg: AsrConfig | None = None,
    spoofCfg: SpoofConfig | None = None,
    llmCfg: LlmConfig | None = None,
    ttsCfg: TtsConfig | None = None,
    runTts: bool = True,
) -> RiskReport:
    asrCfg = asrCfg or AsrConfig()
    spoofCfg = spoofCfg or SpoofConfig()
    llmCfg = llmCfg or LlmConfig()
    ttsCfg = ttsCfg or TtsConfig()

    audioPath = Path(audioPath)
    outDir = Path(outDir)
    outDir.mkdir(parents=True, exist_ok=True)

    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    clip = ingest(audioPath)
    timings["ingestion"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    transcript = transcribe(clip, asrCfg)
    timings["asr"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    spoof = detectSpoof(clip, spoofCfg)
    timings["antiSpoof"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    tactics = classifyTactics(transcript, llmCfg)
    timings["tacticClassification"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    risk = synthesizeRisk(transcript, spoof, tactics, llmCfg)
    timings["riskSynthesis"] = time.perf_counter() - t0

    audio_meta = AudioMeta(
        sourcePath=audioPath,
        durationSec=clip.durationSec,
        sampleRate=clip.sampleRate,
    )

    briefing_path: Path | None = None
    if runTts:
        partial_report = RiskReport(
            version="1.0.0",
            audio=audio_meta,
            transcript=transcript,
            spoof=spoof,
            tactics=tactics,
            risk=risk,
            briefingAudioPath=None,
            timings=timings,
        )
        t0 = time.perf_counter()
        briefing_path = narrateBriefing(partial_report, ttsCfg, outDir)
        timings["ttsBriefing"] = time.perf_counter() - t0

    timings["total"] = sum(v for k, v in timings.items() if k != "total")

    report = RiskReport(
        version="1.0.0",
        audio=audio_meta,
        transcript=transcript,
        spoof=spoof,
        tactics=tactics,
        risk=risk,
        briefingAudioPath=briefing_path,
        timings=timings,
    )

    saveReport(report, outDir / "report.json")
    return report
