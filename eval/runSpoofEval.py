"""Phase 3 T3.2 — anti-spoof accuracy / F1 / EER on the manifest.

Usage (Colab GPU):
    python -m eval.runSpoofEval --manifest eval/out/spoof/manifest.csv \
                                --out eval/out/spoof/

Outputs:
    eval/out/spoof/metrics.csv          — per-subset + overall metrics
    eval/out/spoof/confusion_matrix.png — confusion matrix heatmap
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Pure helper functions (unit-tested locally)
# ---------------------------------------------------------------------------

def computeMetrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    """Compute accuracy, precision, recall, F1 for binary classification."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support  # noqa: PLC0415

    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "accuracy": acc,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }


def computeEer(y_true: list[int], scores: list[float]) -> float:
    """Compute Equal Error Rate from ground-truth labels and continuous scores.

    Convention: label 1 = fake/positive, 0 = real/negative.
    EER is the threshold where FPR == FNR.
    """
    from sklearn.metrics import roc_curve  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    if len(set(y_true)) < 2:
        return 0.0

    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
    fnr = 1.0 - tpr
    # find threshold index closest to EER
    idx = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def loadManifest(path: Path) -> Any:
    """Load manifest CSV → DataFrame with columns path, label, source."""
    import pandas as pd  # noqa: PLC0415

    df = pd.read_csv(path)
    return df


# ---------------------------------------------------------------------------
# Eval driver (model calls — runs on Colab GPU)
# ---------------------------------------------------------------------------

def _run_eval(manifest_path: Path, out_dir: Path) -> None:
    import numpy as np
    import pandas as pd
    import librosa
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    from vishguard.antiSpoof import detectSpoof
    from vishguard.loadAudio import ingest
    from vishguard.types import SpoofConfig

    cfg = SpoofConfig()
    df = loadManifest(manifest_path)

    y_true: list[int] = []
    y_scores: list[float] = []
    y_pred: list[int] = []
    sources: list[str] = []

    for _, row in df.iterrows():
        clip = ingest(row["path"])
        verdict = detectSpoof(clip, cfg)
        label_int = 1 if row["label"] == "fake" else 0
        pred_int = 1 if verdict.pSynthetic >= 0.5 else 0
        y_true.append(label_int)
        y_scores.append(verdict.pSynthetic)
        y_pred.append(pred_int)
        sources.append(row["source"])

    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for subset in sorted(set(sources)) + ["all"]:
        if subset == "all":
            mask = list(range(len(y_true)))
        else:
            mask = [i for i, s in enumerate(sources) if s == subset]
        yt = [y_true[i] for i in mask]
        yp = [y_pred[i] for i in mask]
        ys = [y_scores[i] for i in mask]
        m = computeMetrics(yt, yp)
        m["eer"] = computeEer(yt, ys)
        m["subset"] = subset
        m["n"] = len(mask)
        rows.append(m)

    pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["real", "fake"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Anti-spoof confusion matrix")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    print(f"Saved confusion matrix → {out_dir / 'confusion_matrix.png'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run anti-spoof evaluation.")
    parser.add_argument("--manifest", type=Path, default=Path("eval/out/spoof/manifest.csv"))
    parser.add_argument("--out", type=Path, default=Path("eval/out/spoof/"))
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"ERROR: manifest not found: {args.manifest}")
        return 1

    _run_eval(args.manifest, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
