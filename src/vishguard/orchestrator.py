"""Pipeline orchestrator — wires every stage, captures per-stage latency."""
from __future__ import annotations

from pathlib import Path

from .types import RiskReport


def runPipeline(audioPath: Path, outDir: Path) -> RiskReport:
    raise NotImplementedError("T2.7 orchestrator.runPipeline — see docs/TASKS.md")
