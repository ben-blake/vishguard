"""Pydantic schema validation for report.json artifacts.

Single source of truth for the JSON contract documented in
docs/ARCHITECTURE.md §4. Every eval script must validate through here.
"""
from __future__ import annotations

from pathlib import Path

from .types import RiskReport


def validateReport(reportPath: Path) -> RiskReport:
    raise NotImplementedError("T2.7 reportSchema.validateReport — see docs/TASKS.md")
