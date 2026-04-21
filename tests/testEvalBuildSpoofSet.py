"""Unit tests for eval/buildSpoofSet.py helper functions (T3.1)."""
import csv
import io
from pathlib import Path

import pytest

# Import helpers — will raise ImportError until T3.1 is implemented.
from eval.buildSpoofSet import buildManifest, validateManifestColumns


# ---------------------------------------------------------------------------
# buildManifest
# ---------------------------------------------------------------------------

_ROWS = [
    {"path": "/a/real1.wav", "label": "real", "source": "librispeech"},
    {"path": "/a/real2.wav", "label": "real", "source": "librispeech"},
    {"path": "/b/fake1.wav", "label": "fake", "source": "speecht5"},
]


def test_build_manifest_returns_required_columns():
    df = buildManifest(_ROWS)
    assert set(df.columns) >= {"path", "label", "source"}


def test_build_manifest_row_count():
    df = buildManifest(_ROWS)
    assert len(df) == 3


def test_build_manifest_labels_preserved():
    df = buildManifest(_ROWS)
    assert list(df["label"]) == ["real", "real", "fake"]


def test_build_manifest_source_values():
    df = buildManifest(_ROWS)
    assert set(df["source"].unique()) == {"librispeech", "speecht5"}


# ---------------------------------------------------------------------------
# validateManifestColumns
# ---------------------------------------------------------------------------

def test_validate_columns_passes_for_good_df():
    import pandas as pd  # local import so test can be collected without pandas
    df = pd.DataFrame(_ROWS)
    assert validateManifestColumns(df) is True


def test_validate_columns_fails_missing_label():
    import pandas as pd
    df = pd.DataFrame([{"path": "/x.wav", "source": "librispeech"}])
    assert validateManifestColumns(df) is False


def test_validate_columns_fails_invalid_label_value():
    import pandas as pd
    df = pd.DataFrame([{"path": "/x.wav", "label": "unknown", "source": "librispeech"}])
    assert validateManifestColumns(df) is False
