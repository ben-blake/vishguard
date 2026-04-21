"""Phase 3 T3.4 — tactic classifier prompt v1 vs v2 macro-F1.

Usage (Colab GPU):
    python -m eval.runTacticEval --corpus eval/data/tactics.jsonl \
                                 --out eval/out/tactics/

Outputs:
    eval/out/tactics/metrics.csv   — per-label F1 + macro for v1 and v2
"""
from __future__ import annotations

import argparse
from pathlib import Path

from vishguard.types import TACTIC_TAXONOMY


# ---------------------------------------------------------------------------
# Pure helper functions (unit-tested locally)
# ---------------------------------------------------------------------------

def computePerLabelF1(label: str, y_true: list[set], y_pred: list[set]) -> float:
    """Binary F1 for a single label across multi-label examples."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if label in t and label in p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if label not in t and label in p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if label in t and label not in p)
    if tp + fp + fn == 0:
        return 0.0
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    if prec + rec == 0.0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def computeMacroF1(y_true: list[set], y_pred: list[set], labels: tuple[str, ...]) -> float:
    """Macro-averaged F1 across all provided labels."""
    per_label = [computePerLabelF1(lbl, y_true, y_pred) for lbl in labels]
    return sum(per_label) / len(per_label)


# ---------------------------------------------------------------------------
# Eval driver (model calls — runs on Colab GPU)
# ---------------------------------------------------------------------------

def _run_eval(corpus_path: Path, out_dir: Path) -> None:
    import pandas as pd
    from vishguard.tacticClassifier import classifyTactics  # noqa: PLC0415
    from vishguard.types import LlmConfig, Transcript, TranscriptSegment  # noqa: PLC0415

    from eval.buildTacticSet import loadCorpus as _load  # noqa: PLC0415

    examples = _load(corpus_path)
    y_true: list[set] = [set(ex["labels"]) for ex in examples]

    def _make_transcript(text: str) -> Transcript:
        seg = TranscriptSegment(startSec=0.0, endSec=30.0, text=text)
        return Transcript(fullText=text, segments=(seg,), languageCode="en", modelId="eval-corpus")

    rows: list[dict] = []
    for variant in ("v1", "v2", "v3", "v4"):
        cfg = LlmConfig(promptVariant=variant)
        y_pred: list[set] = []
        for i, ex in enumerate(examples):
            try:
                tactics = classifyTactics(_make_transcript(ex["text"]), cfg)
            except Exception as exc:
                print(f"  [WARN] example {i} failed ({exc}); counting as no prediction")
                tactics = ()
            y_pred.append({t.label for t in tactics})

        macro = computeMacroF1(y_true, y_pred, TACTIC_TAXONOMY)
        row: dict = {"variant": variant, "macro_f1": round(macro, 4)}
        for lbl in TACTIC_TAXONOMY:
            row[f"f1_{lbl}"] = round(computePerLabelF1(lbl, y_true, y_pred), 4)
        rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metrics.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nSaved → {out_dir / 'metrics.csv'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run tactic-classification evaluation.")
    parser.add_argument("--corpus", type=Path, default=Path("eval/data/tactics.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("eval/out/tactics/"))
    args = parser.parse_args()

    if not args.corpus.exists():
        print(f"ERROR: corpus not found: {args.corpus}")
        return 1

    _run_eval(args.corpus, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
