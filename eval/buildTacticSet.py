"""Phase 3 T3.3 — validate and report on the tactic-classification corpus.

The corpus lives at eval/data/tactics.jsonl (50 hand-verified scripts).
This script validates it and prints a coverage report.

Usage:
    python -m eval.buildTacticSet [--corpus eval/data/tactics.jsonl]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vishguard.types import TACTIC_TAXONOMY

_DEFAULT_CORPUS = Path(__file__).parent / "data" / "tactics.jsonl"


def loadCorpus(path: Path) -> list[dict]:
    """Load a JSONL file and return a list of entry dicts."""
    entries: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def validateEntry(entry: dict, taxonomy: tuple[str, ...]) -> list[str]:
    """Return a list of validation error strings (empty = valid)."""
    errors: list[str] = []
    if not entry.get("text", "").strip():
        errors.append("text is empty or missing")
    labels = entry.get("labels")
    if labels is None:
        errors.append("labels field missing")
    elif not isinstance(labels, list) or len(labels) == 0:
        errors.append("labels must be a non-empty list")
    else:
        for lbl in labels:
            if lbl not in taxonomy:
                errors.append(f"unknown label: {lbl!r}")
    return errors


def validateCorpus(examples: list[dict], taxonomy: tuple[str, ...]) -> list[str]:
    """Validate all entries; return list of error strings (empty = clean)."""
    all_errors: list[str] = []
    for i, entry in enumerate(examples):
        for err in validateEntry(entry, taxonomy):
            all_errors.append(f"entry {i}: {err}")
    return all_errors


def _coverage_report(examples: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {t: 0 for t in TACTIC_TAXONOMY}
    for entry in examples:
        for lbl in entry.get("labels", []):
            if lbl in counts:
                counts[lbl] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate tactic corpus.")
    parser.add_argument("--corpus", type=Path, default=_DEFAULT_CORPUS)
    args = parser.parse_args()

    if not args.corpus.exists():
        print(f"ERROR: corpus file not found: {args.corpus}")
        return 1

    examples = loadCorpus(args.corpus)
    errors = validateCorpus(examples, TACTIC_TAXONOMY)

    print(f"Corpus: {args.corpus}")
    print(f"Entries: {len(examples)}")

    if errors:
        print(f"VALIDATION ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  {e}")
        return 1

    print("Validation: PASS")

    counts = _coverage_report(examples)
    print("\nLabel coverage:")
    for label, count in sorted(counts.items(), key=lambda x: -x[1]):
        bar = "#" * count
        print(f"  {label:<25} {count:3d}  {bar}")

    missing = [lbl for lbl, n in counts.items() if n == 0]
    if missing:
        print(f"\nWARNING: labels with zero examples: {missing}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
