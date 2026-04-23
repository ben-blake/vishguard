"""Streamlit report page — renders a RiskReport for the demo UI."""
from __future__ import annotations

from typing import Any

from ..types import RiskReport, Tactic

_BAND_COLORS: dict[str, str] = {
    "low": "#2ecc71",
    "medium": "#f39c12",
    "high": "#e67e22",
    "critical": "#e74c3c",
}
_FALLBACK_COLOR = "#95a5a6"


def bandColor(band: str) -> str:
    return _BAND_COLORS.get(band, _FALLBACK_COLOR)


def spoofLabel(pSynthetic: float, threshold: float = 0.5) -> str:
    return "Synthetic" if pSynthetic >= threshold else "Real"


def formatPct(value: float) -> str:
    return f"{round(value * 100)}%"


def tacticRows(tactics: tuple[Tactic, ...]) -> list[dict[str, Any]]:
    return [{"label": t.label, "confidence": t.confidence} for t in tactics]


def timingsTable(timings: dict[str, float]) -> list[dict[str, Any]]:
    return [{"stage": k, "duration_s": v} for k, v in timings.items()]


def renderReport(report: RiskReport) -> None:
    import streamlit as st

    color = bandColor(report.risk.band)
    st.markdown(
        f"<h1 style='color:{color}'>Risk Score: {report.risk.score} "
        f"({report.risk.band.upper()})</h1>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Synthetic probability", formatPct(report.spoof.pSynthetic))
        st.caption(f"Verdict: {spoofLabel(report.spoof.pSynthetic)} — {report.spoof.rationale}")
    with col2:
        st.metric("Audio duration", f"{report.audio.durationSec:.1f} s")
        st.caption(f"Source: {report.audio.sourcePath.name}")

    with st.expander("Transcript", expanded=True):
        st.write(report.transcript.fullText)

    with st.expander("Detected tactics"):
        rows = tacticRows(report.tactics)
        if rows:
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("No malicious tactics detected.")

    with st.expander("Risk reasoning"):
        st.write(report.risk.reasoning)

    if report.briefingAudioPath and report.briefingAudioPath.exists():
        with st.expander("Spoken briefing", expanded=True):
            st.audio(str(report.briefingAudioPath))

    with st.expander("Stage timings"):
        st.dataframe(timingsTable(report.timings), use_container_width=True)
