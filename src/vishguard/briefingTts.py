"""SpeechT5 TTS briefing stage — RiskReport -> WAV path."""
from __future__ import annotations

from pathlib import Path

import torch

from .types import RiskReport, TtsConfig

_SAMPLE_RATE = 16_000
_SPEAKER_IDX = 7306  # deterministic speaker from cmu-arctic-xvectors


def _buildBriefingText(report: RiskReport) -> str:
    tactic_labels = [t.label for t in report.tactics if t.label != "benign"]
    tactics_str = ", ".join(tactic_labels) if tactic_labels else "none detected"
    return (
        f"VishGuard alert. Risk score {report.risk.score} out of 100, "
        f"level {report.risk.band}. "
        f"Tactics identified: {tactics_str}. "
        f"{report.risk.reasoning}"
    )


def narrateBriefing(report: RiskReport, cfg: TtsConfig, outDir: Path) -> Path:
    import soundfile as sf
    from transformers import (
        SpeechT5ForTextToSpeech,
        SpeechT5HifiGan,
        SpeechT5Processor,
    )

    text = _buildBriefingText(report)

    # Separate from_pretrained and .to() so Pylance can resolve the instance type.
    processor = SpeechT5Processor.from_pretrained(cfg.modelId)
    model = SpeechT5ForTextToSpeech.from_pretrained(cfg.modelId)
    model = model.to(cfg.device)  # type: ignore[arg-type]
    vocoder = SpeechT5HifiGan.from_pretrained(cfg.vocoderId)
    vocoder = vocoder.to(cfg.device)  # type: ignore[arg-type]

    import zipfile
    import numpy as np
    from pathlib import Path as _Path
    from huggingface_hub import snapshot_download

    xvec_repo = snapshot_download(
        repo_id=cfg.speakerEmbeddingId,
        repo_type="dataset",
        ignore_patterns=["*.py"],
    )
    zip_path = _Path(xvec_repo) / "spkrec-xvect.zip"
    extract_dir = _Path("/tmp/cmu_arctic_xvect")
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    npy_files = sorted(extract_dir.rglob("*.npy"))
    pt_files = sorted(extract_dir.rglob("*.pt"))

    if npy_files:
        xvec = np.load(npy_files[_SPEAKER_IDX % len(npy_files)])
        speaker_embedding = torch.tensor(xvec).unsqueeze(0).float().to(cfg.device)
    elif pt_files:
        data = torch.load(pt_files[0], map_location="cpu")
        if isinstance(data, torch.Tensor):
            row = data[min(_SPEAKER_IDX, data.shape[0] - 1)]
        else:
            row = list(data.values())[0]
        speaker_embedding = row.unsqueeze(0).float().to(cfg.device)
    else:
        speaker_embedding = torch.nn.functional.normalize(
            torch.randn(1, 512), dim=-1
        ).to(cfg.device)

    # Processor __call__ is typed Optional in some stubs; split to allow .to().
    encoding = processor(text=text, return_tensors="pt")
    inputs = encoding.to(cfg.device)  # type: ignore[union-attr]

    with torch.no_grad():
        # generate_speech returns FloatTensor; stubs may narrow to tuple when
        # return_output_lengths is not set — annotate explicitly.
        speech: torch.Tensor = model.generate_speech(  # type: ignore[assignment]
            inputs["input_ids"], speaker_embedding, vocoder=vocoder  # type: ignore[arg-type]
        )

    outDir = Path(outDir)
    outDir.mkdir(parents=True, exist_ok=True)
    out_path = outDir / "briefing.wav"
    sf.write(str(out_path), speech.cpu().numpy(), samplerate=_SAMPLE_RATE)
    return out_path
