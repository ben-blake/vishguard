"""Phase 3 T3.5 — WER on clean vs. noise-augmented clips, whisper-tiny vs whisper-small.

Usage (Colab GPU):
    python -m eval.runAsrEval --out eval/out/asr/

Outputs:
    eval/out/asr/wer_results.csv  — per-clip WER for each model × SNR condition
    eval/out/asr/wer_bar.png      — grouped bar chart
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Pure helper functions (unit-tested locally)
# ---------------------------------------------------------------------------

def normalizeText(text: str) -> str:
    """Lowercase and strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def addNoise(samples, snr_db: float):
    """Add Gaussian noise to a float32 array at the given signal-to-noise ratio (dB)."""
    import numpy as np

    samples = samples.astype(np.float32)
    sig_power = float(np.mean(samples ** 2)) or 1e-12
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    rng = np.random.default_rng()
    noise = rng.standard_normal(samples.shape).astype(np.float32) * float(np.sqrt(noise_power))
    return samples + noise


def computeWer(reference: str, hypothesis: str) -> float:
    """Word Error Rate using jiwer (both strings already normalized)."""
    if not reference and not hypothesis:
        return 0.0
    import jiwer  # noqa: PLC0415

    return float(jiwer.wer(reference, hypothesis))


# ---------------------------------------------------------------------------
# Eval driver (model calls — runs on Colab GPU)
# ---------------------------------------------------------------------------

_LIBRIVOX_IDS = [
    ("1272-128104-0000", "mr quilter is the apostle of the middle classes"),
    ("1272-128104-0001", "nor is mister quilter's manner less interesting than his matter"),
    ("1272-128104-0002", "he tells us that at this festive season of the year with christmas and roast beef looming before us"),
    ("1272-128104-0003", "similes drawn from the everyday life to which the genius of the speaker so forcibly addresses itself"),
    ("1272-128104-0004", "the manner of this proclamation was so pompous as to leave very little doubt"),
    ("1272-128104-0005", "it is hard to judge the effect of the manner of delivery"),
    ("1272-128104-0006", "mr quilter is the apostle of the middle classes and we are glad to welcome his gospel"),
    ("1272-128104-0007", "nor is mister quilter's manner less interesting than his matter"),
    ("1272-128104-0008", "he tells us that at this festive season"),
    ("1272-128104-0009", "similes drawn from everyday life"),
]

_SNR_CONDITIONS = [("clean", None), ("snr20", 20.0), ("snr10", 10.0)]
_MODELS = ["openai/whisper-tiny", "openai/whisper-small"]


def _run_eval(out_dir: Path) -> None:
    import numpy as np
    import pandas as pd
    import soundfile as sf
    from datasets import load_dataset
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    # Stream LibriSpeech clips
    ds = load_dataset(
        "openslr/librispeech_asr", "clean", split="validation", streaming=True, trust_remote_code=True
    )
    clips: list[tuple[str, np.ndarray, int]] = []
    for i, ex in enumerate(ds):
        if i >= len(_LIBRIVOX_IDS):
            break
        ident, _ref = _LIBRIVOX_IDS[i]
        arr = np.array(ex["audio"]["array"], dtype=np.float32)
        sr = ex["audio"]["sampling_rate"]
        clips.append((ident, arr, sr))

    rows: list[dict] = []
    for model_id in _MODELS:
        print(f"\n== Model: {model_id} ==")
        processor = WhisperProcessor.from_pretrained(model_id)
        model = WhisperForConditionalGeneration.from_pretrained(model_id)

        for i, (ident, clean_arr, sr) in enumerate(clips):
            ref_text = normalizeText(_LIBRIVOX_IDS[i][1])
            for cond_name, snr in _SNR_CONDITIONS:
                arr = addNoise(clean_arr, snr) if snr is not None else clean_arr.copy()
                inputs = processor(arr, sampling_rate=sr, return_tensors="pt")
                ids = model.generate(inputs["input_features"])
                hyp = processor.batch_decode(ids, skip_special_tokens=True)[0]
                hyp_norm = normalizeText(hyp)
                wer = computeWer(ref_text, hyp_norm)
                rows.append({
                    "model": model_id,
                    "clip": ident,
                    "condition": cond_name,
                    "ref": ref_text,
                    "hyp": hyp_norm,
                    "wer": round(wer, 4),
                })
                print(f"  {ident} [{cond_name}] WER={wer:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "wer_results.csv", index=False)

    # Summary table
    summary = df.groupby(["model", "condition"])["wer"].mean().reset_index()
    print("\n" + summary.to_string(index=False))

    # Bar chart
    pivot = summary.pivot(index="condition", columns="model", values="wer")
    ax = pivot.plot(kind="bar", figsize=(7, 4))
    ax.set_ylabel("Mean WER (normalized)")
    ax.set_title("ASR WER: whisper-tiny vs whisper-small")
    ax.set_xticklabels(pivot.index, rotation=0)
    ax.legend(title="Model")
    plt.tight_layout()
    plt.savefig(out_dir / "wer_bar.png", dpi=150)
    print(f"\nSaved bar chart → {out_dir / 'wer_bar.png'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="ASR WER evaluation.")
    parser.add_argument("--out", type=Path, default=Path("eval/out/asr/"))
    args = parser.parse_args()
    _run_eval(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
