"""Unit tests for eval/buildTacticSet.py helper functions (T3.3)."""
import json
import tempfile
from pathlib import Path

import pytest

from eval.buildTacticSet import validateEntry, validateCorpus, loadCorpus
from vishguard.types import TACTIC_TAXONOMY


_GOOD_ENTRY = {
    "text": "Hello, this is your bank calling to verify your account.",
    "labels": ["credential_harvesting", "impersonation"],
    "notes": "classic bank spoof",
}


# ---------------------------------------------------------------------------
# validateEntry
# ---------------------------------------------------------------------------

def test_validate_entry_good():
    errors = validateEntry(_GOOD_ENTRY, TACTIC_TAXONOMY)
    assert errors == []


def test_validate_entry_missing_text():
    bad = {**_GOOD_ENTRY, "text": ""}
    errors = validateEntry(bad, TACTIC_TAXONOMY)
    assert any("text" in e for e in errors)


def test_validate_entry_missing_labels():
    bad = {"text": "hello", "notes": "x"}
    errors = validateEntry(bad, TACTIC_TAXONOMY)
    assert any("label" in e for e in errors)


def test_validate_entry_unknown_label():
    bad = {**_GOOD_ENTRY, "labels": ["not_in_taxonomy"]}
    errors = validateEntry(bad, TACTIC_TAXONOMY)
    assert any("not_in_taxonomy" in e for e in errors)


def test_validate_entry_empty_labels_list():
    bad = {**_GOOD_ENTRY, "labels": []}
    errors = validateEntry(bad, TACTIC_TAXONOMY)
    assert errors  # at least one error


# ---------------------------------------------------------------------------
# validateCorpus
# ---------------------------------------------------------------------------

def test_validate_corpus_no_errors():
    corpus = [_GOOD_ENTRY] * 5
    errors = validateCorpus(corpus, TACTIC_TAXONOMY)
    assert errors == []


def test_validate_corpus_reports_all_bad():
    bad = {"text": "", "labels": ["bad_label"], "notes": ""}
    errors = validateCorpus([bad, bad], TACTIC_TAXONOMY)
    assert len(errors) >= 2


# ---------------------------------------------------------------------------
# loadCorpus
# ---------------------------------------------------------------------------

def test_load_corpus_returns_list():
    entries = [_GOOD_ENTRY, _GOOD_ENTRY]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
        path = Path(f.name)
    result = loadCorpus(path)
    assert isinstance(result, list)
    assert len(result) == 2


def test_load_corpus_preserves_fields():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps(_GOOD_ENTRY) + "\n")
        path = Path(f.name)
    result = loadCorpus(path)
    assert result[0]["labels"] == _GOOD_ENTRY["labels"]
