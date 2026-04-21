"""Phase 3 T3.1 — build anti-spoof eval manifest from LibriSpeech + SpeechT5.

Usage (Colab GPU):
    python -m eval.buildSpoofSet --out eval/out/spoof/manifest.csv

The manifest CSV has columns: path, label (real|fake), source.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

_VALID_LABELS = {"real", "fake"}
_VALID_SOURCES = {"librispeech", "speecht5", "asvspoof"}


def buildManifest(rows: list[dict[str, str]]) -> Any:
    """Convert a list of {path, label, source} dicts into a DataFrame."""
    import pandas as pd  # noqa: PLC0415

    return pd.DataFrame(rows, columns=["path", "label", "source"])


def validateManifestColumns(df: Any) -> bool:
    """Return True iff df has required columns and only valid label values."""
    required = {"path", "label", "source"}
    if not required.issubset(set(df.columns)):
        return False
    if not set(df["label"].unique()).issubset(_VALID_LABELS):
        return False
    return True


def _stream_librispeech_clips(n: int, out_dir: Path) -> list[dict[str, str]]:
    """Download n real clips from LibriSpeech dev-clean via HuggingFace datasets."""
    from datasets import load_dataset  # noqa: PLC0415
    import soundfile as sf  # noqa: PLC0415

    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(
        "openslr/librispeech_asr",
        "clean",
        split="validation",
        streaming=True,
        trust_remote_code=True,
    )
    rows: list[dict[str, str]] = []
    for i, ex in enumerate(ds):
        if i >= n:
            break
        audio = ex["audio"]
        samples = audio["array"]
        sr = audio["sampling_rate"]
        wav_path = out_dir / f"real_{i:04d}.wav"
        sf.write(str(wav_path), samples, sr)
        rows.append({"path": str(wav_path), "label": "real", "source": "librispeech"})
    return rows


def _generate_speecht5_clips(n: int, out_dir: Path) -> list[dict[str, str]]:
    """Generate n fake clips via SpeechT5 TTS."""
    import torch  # noqa: PLC0415
    import soundfile as sf  # noqa: PLC0415
    from datasets import load_dataset  # noqa: PLC0415
    from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor  # noqa: PLC0415

    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
    embeds_ds = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_emb = torch.tensor(embeds_ds[7306]["xvector"]).unsqueeze(0).to(device)

    scripts = [
        "Your account has been suspended. Please call us immediately to avoid legal action.",
        "Congratulations, you have won a prize. Please verify your details to claim it.",
        "This is an urgent security alert. Your password has been compromised.",
        "We need to verify your identity. Please provide your social security number.",
        "Your computer is infected with a virus. Call our tech support line now.",
        "This is Agent Miller from the IRS. You owe back taxes and must pay today.",
        "Your bank account has unusual activity. Press one to speak to an agent.",
        "You qualify for a low interest loan. Provide your account details to proceed.",
        "A warrant has been issued for your arrest. Call this number immediately.",
        "We are calling from Microsoft. Your Windows license has expired.",
        "Please confirm your credit card number to complete your order.",
        "You have been selected for an exclusive investment opportunity.",
        "Your package delivery requires identity verification.",
        "Our records show you claimed a reward but did not collect it.",
        "This is a final notice before we take legal action against you.",
        "Your social security number has been suspended due to suspicious activity.",
        "Press 2 to speak with a senior tax investigator immediately.",
        "We detected a large wire transfer from your account. Verify now.",
        "Your subscription will renew at a higher rate unless you call us.",
        "Complete this brief survey to receive your complimentary gift card.",
    ]

    rows: list[dict[str, str]] = []
    for i in range(n):
        script = scripts[i % len(scripts)]
        inputs = processor(text=script, return_tensors="pt").to(device)
        with torch.no_grad():
            speech = model.generate_speech(inputs["input_ids"], speaker_emb, vocoder=vocoder)
        wav_path = out_dir / f"fake_{i:04d}.wav"
        sf.write(str(wav_path), speech.cpu().numpy(), 16000)
        rows.append({"path": str(wav_path), "label": "fake", "source": "speecht5"})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build anti-spoof eval manifest.")
    parser.add_argument("--n-real", type=int, default=50, help="real clips from LibriSpeech")
    parser.add_argument("--n-fake", type=int, default=50, help="fake clips from SpeechT5")
    parser.add_argument("--out", type=Path, default=Path("eval/out/spoof/manifest.csv"))
    args = parser.parse_args()

    audio_dir = args.out.parent / "audio"
    print(f"Downloading {args.n_real} real clips …")
    real_rows = _stream_librispeech_clips(args.n_real, audio_dir / "real")
    print(f"Generating {args.n_fake} synthetic clips …")
    fake_rows = _generate_speecht5_clips(args.n_fake, audio_dir / "fake")

    all_rows = real_rows + fake_rows
    df = buildManifest(all_rows)
    if not validateManifestColumns(df):
        print("ERROR: manifest validation failed")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
