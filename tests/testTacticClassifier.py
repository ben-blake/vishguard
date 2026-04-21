"""T2.4: Tests for tacticClassifier — JSON parsing, taxonomy enforcement, retry."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from vishguard.tacticClassifier import _extract_json, _parse_tactics
from vishguard.types import TACTIC_TAXONOMY, LlmConfig, Tactic, Transcript, TranscriptSegment


def _make_transcript(text: str) -> Transcript:
    seg = TranscriptSegment(startSec=0.0, endSec=5.0, text=text)
    return Transcript(fullText=text, segments=(seg,), languageCode="en", modelId="test")


class TestExtractJson:
    def test_extracts_plain_array(self) -> None:
        text = '[{"label": "authority", "confidence": 0.9, "evidenceSpans": []}]'
        result = _extract_json(text)
        assert isinstance(result, list)
        assert result[0]["label"] == "authority"

    def test_extracts_array_with_surrounding_prose(self) -> None:
        text = (
            'Here is my analysis:\n'
            '[{"label": "benign", "confidence": 0.99, "evidenceSpans": []}]\n'
            'Done.'
        )
        result = _extract_json(text)
        assert result[0]["label"] == "benign"

    def test_raises_on_no_array(self) -> None:
        with pytest.raises(ValueError):
            _extract_json("No array here at all.")

    def test_raises_on_malformed_json(self) -> None:
        with pytest.raises((ValueError, json.JSONDecodeError)):
            _extract_json("[{bad json}")


class TestParseTactics:
    def test_filters_invalid_labels(self) -> None:
        raw = [
            {"label": "authority", "confidence": 0.9, "evidenceSpans": []},
            {"label": "not_a_real_tactic", "confidence": 0.5, "evidenceSpans": []},
        ]
        tactics = _parse_tactics(raw)
        assert len(tactics) == 1
        assert tactics[0].label == "authority"

    def test_all_taxonomy_labels_accepted(self) -> None:
        raw = [
            {"label": lbl, "confidence": 0.8, "evidenceSpans": []}
            for lbl in TACTIC_TAXONOMY
        ]
        tactics = _parse_tactics(raw)
        assert len(tactics) == len(TACTIC_TAXONOMY)

    def test_returns_tuple_of_tactic(self) -> None:
        raw = [{"label": "urgency", "confidence": 0.7, "evidenceSpans": ["act now"]}]
        tactics = _parse_tactics(raw)
        assert isinstance(tactics, tuple)
        assert isinstance(tactics[0], Tactic)

    def test_evidence_spans_are_tuple(self) -> None:
        raw = [{"label": "urgency", "confidence": 0.7, "evidenceSpans": ["act now", "today only"]}]
        tactics = _parse_tactics(raw)
        assert isinstance(tactics[0].evidenceSpans, tuple)

    def test_empty_raw_returns_empty_tuple(self) -> None:
        assert _parse_tactics([]) == ()


class TestClassifyTactics:
    def test_returns_tuple_on_valid_json_output(self) -> None:
        from vishguard.tacticClassifier import classifyTactics

        transcript = _make_transcript("This is the IRS. Pay $2400 in gift cards immediately.")
        cfg = LlmConfig(modelId="test-llm", device="cpu", loadIn4Bit=False)

        mock_output = json.dumps([
            {"label": "authority", "confidence": 0.95, "evidenceSpans": ["This is the IRS"]},
            {"label": "financial_manipulation", "confidence": 0.9, "evidenceSpans": ["Pay $2400"]},
        ])

        with patch("vishguard.tacticClassifier._call_llm", return_value=mock_output):
            tactics = classifyTactics(transcript, cfg)

        assert isinstance(tactics, tuple)
        assert all(isinstance(t, Tactic) for t in tactics)
        assert {t.label for t in tactics} <= set(TACTIC_TAXONOMY)

    def test_retries_on_parse_failure(self) -> None:
        from vishguard.tacticClassifier import classifyTactics

        transcript = _make_transcript("Hello this is a test call.")
        cfg = LlmConfig(modelId="test-llm", device="cpu", loadIn4Bit=False)

        valid_output = '[{"label": "benign", "confidence": 0.99, "evidenceSpans": []}]'

        with patch("vishguard.tacticClassifier._call_llm") as mock_llm:
            mock_llm.side_effect = ["not valid json at all {{{", valid_output]
            tactics = classifyTactics(transcript, cfg)

        assert mock_llm.call_count == 2
        assert tactics[0].label == "benign"

    def test_uses_v2_prompt_by_default(self) -> None:
        from vishguard.tacticClassifier import classifyTactics
        from vishguard.promptLibrary import tacticPromptV2

        transcript = _make_transcript("Hello world.")
        cfg = LlmConfig(modelId="test-llm", device="cpu", loadIn4Bit=False, promptVariant="v2")

        with patch("vishguard.tacticClassifier._call_llm", return_value='[{"label":"benign","confidence":0.9,"evidenceSpans":[]}]') as mock_llm:
            classifyTactics(transcript, cfg)

        called_messages = mock_llm.call_args[0][0]
        v2_messages = tacticPromptV2(transcript.fullText)
        assert called_messages[0]["content"] == v2_messages[0]["content"]
