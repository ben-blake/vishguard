"""Streamlit entry point for VishGuard (T4.1).

Run with:
    streamlit run app.py
or with whisper-tiny for fast local demo:
    streamlit run app.py -- --asr-model openai/whisper-tiny
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
import torch

from vishguard.orchestrator import runPipeline
from vishguard.types import AsrConfig, LlmConfig, SpoofConfig, TtsConfig, Transcript
from vishguard.ui.pageReport import renderReport

st.set_page_config(
    page_title="VishGuard — Vishing Call Analyzer",
    page_icon="🔍",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading models into GPU…")
def _prewarm(device: str) -> None:
    """Load all pipeline models into module-level caches on first page load.

    @st.cache_resource runs once per Streamlit server lifetime, so subsequent
    analysis clicks skip the disk→VRAM loading step entirely.
    """
    import numpy as np
    import soundfile as sf
    import tempfile
    from vishguard.loadAudio import ingest
    from vishguard.asrWhisper import transcribe
    from vishguard.antiSpoof import detectSpoof
    from vishguard.tacticClassifier import classifyTactics

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, np.zeros(16_000, dtype=np.float32), 16_000)
        dummy_path = Path(f.name)

    clip = ingest(dummy_path)
    transcribe(clip, AsrConfig(modelId="openai/whisper-small", device=device))
    detectSpoof(clip, SpoofConfig(device=device))
    classifyTactics(
        Transcript(fullText="hello", durationSec=1.0, modelId="openai/whisper-small"),
        LlmConfig(device=device, loadIn4Bit=(device == "cuda")),
    )


def _sidebar_config() -> tuple[AsrConfig, SpoofConfig, LlmConfig, TtsConfig, bool]:
    st.sidebar.header("Pipeline settings")

    asr_model = st.sidebar.selectbox(
        "ASR model",
        ["openai/whisper-tiny", "openai/whisper-small"],
        index=0,
        help="whisper-tiny is fastest on CPU (~30s for a 60s clip).",
    )
    device = st.sidebar.selectbox(
        "Device", ["cpu", "cuda"],
        index=1 if torch.cuda.is_available() else 0,
    )
    prompt_variant = st.sidebar.selectbox(
        "Tactic prompt", ["v4", "v3", "v2", "v1"], index=0,
        help="v4 gives best macro-F1 (0.604).",
    )
    run_tts = st.sidebar.checkbox(
        "Generate spoken briefing (TTS)",
        value=False,
        help="SpeechT5 TTS — adds ~30s on CPU. Disable for faster demo.",
    )

    asr_cfg = AsrConfig(modelId=asr_model, device=device)
    spoof_cfg = SpoofConfig(device=device)
    llm_cfg = LlmConfig(
        device=device,
        loadIn4Bit=(device == "cuda"),
        promptVariant=prompt_variant,
    )
    tts_cfg = TtsConfig(device=device)
    return asr_cfg, spoof_cfg, llm_cfg, tts_cfg, run_tts


def main() -> None:
    if torch.cuda.is_available():
        _prewarm("cuda")

    st.title("VishGuard — Vishing Call Analyzer")
    st.caption(
        "Upload a phone-call recording to detect synthetic/deepfake voice, "
        "transcribe the call, identify social-engineering tactics, and score overall risk."
    )

    asr_cfg, spoof_cfg, llm_cfg, tts_cfg, run_tts = _sidebar_config()

    uploaded = st.file_uploader(
        "Upload audio file (WAV, MP3, M4A)",
        type=["wav", "mp3", "m4a"],
    )

    if uploaded is None:
        st.info("Upload a recording above to begin analysis.")
        return

    st.audio(uploaded, format=uploaded.type)

    if st.button("Analyze call", type="primary"):
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = Path(tmp) / uploaded.name
            audio_path.write_bytes(uploaded.getvalue())
            out_dir = Path(tmp) / "out"

            with st.spinner("Running pipeline — this may take 30–90 s on CPU…"):
                try:
                    report = runPipeline(
                        audioPath=audio_path,
                        outDir=out_dir,
                        asrCfg=asr_cfg,
                        spoofCfg=spoof_cfg,
                        llmCfg=llm_cfg,
                        ttsCfg=tts_cfg,
                        runTts=run_tts,
                    )
                except Exception as exc:
                    st.error(f"Pipeline error: {exc}")
                    return

            renderReport(report)


if __name__ == "__main__":
    main()
