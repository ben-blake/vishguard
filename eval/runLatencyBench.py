"""Phase 3 T3.6 — end-to-end latency bench on 15 s / 45 s / 90 s clips.

Usage (Colab GPU):
    python -m eval.runLatencyBench --out eval/out/latency/

Outputs:
    eval/out/latency/timings.csv  — per-clip per-stage latency rows
    eval/out/latency/summary.csv  — mean latency per stage aggregated across clips
"""
from __future__ import annotations

import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Pure helper functions (unit-tested locally)
# ---------------------------------------------------------------------------

def summarizeTimings(runs: list[dict]) -> dict:
    """Return mean latency per stage across multiple orchestrator timing dicts."""
    if not runs:
        return {}
    keys = list(runs[0].keys())
    result: dict = {}
    for key in keys:
        vals = [r[key] for r in runs if key in r]
        result[f"mean_{key}_s"] = sum(vals) / len(vals) if vals else 0.0
    return result


def makeTimingRow(clip_path: str, duration_sec: float, timings: dict) -> dict:
    """Build a flat CSV row from orchestrator timings for one clip."""
    row: dict = {"clip_path": clip_path, "duration_sec": duration_sec}
    row.update(timings)
    return row


# ---------------------------------------------------------------------------
# Eval driver (model calls — runs on Colab GPU)
# ---------------------------------------------------------------------------

_DURATIONS_SEC = [15, 45, 90]
_SAMPLE_RATE = 16_000


def _make_synthetic_wav(duration_sec: float, out_path: Path) -> None:
    """Generate a silent WAV file of the given duration at 16 kHz."""
    import numpy as np
    import soundfile as sf

    samples = np.zeros(int(duration_sec * _SAMPLE_RATE), dtype=np.float32)
    sf.write(str(out_path), samples, _SAMPLE_RATE)


def _run_bench(out_dir: Path) -> None:
    import pandas as pd

    from vishguard.orchestrator import runPipeline

    wav_dir = out_dir / "audio"
    wav_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    timing_dicts: list[dict] = []

    for dur in _DURATIONS_SEC:
        wav_path = wav_dir / f"bench_{dur}s.wav"
        _make_synthetic_wav(float(dur), wav_path)
        print(f"Running pipeline on {dur}s clip …")

        report = runPipeline(wav_path, out_dir, runTts=False)
        t = report.timings

        row = makeTimingRow(
            clip_path=str(wav_path),
            duration_sec=float(dur),
            timings=dict(t),
        )
        all_rows.append(row)
        timing_dicts.append(dict(t))
        print(f"  total={t.get('total', '?'):.2f}s  asr={t.get('asr', '?'):.2f}s")

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "timings.csv", index=False)

    summary = summarizeTimings(timing_dicts)
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(out_dir / "summary.csv", index=False)

    print("\nLatency summary (means):")
    for k, v in summary.items():
        print(f"  {k:<22} {v:.3f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end latency benchmark.")
    parser.add_argument("--out", type=Path, default=Path("eval/out/latency/"))
    args = parser.parse_args()
    _run_bench(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
