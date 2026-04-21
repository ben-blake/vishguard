"""Prompt variants for tactic classification and risk synthesis.

Phase 3 A/B tests v1 (bare) vs. v2 (taxonomy + few-shot + JSON schema).
Both variants read TACTIC_TAXONOMY from types.py so the label set cannot
drift between prompts and evaluation.

v1: minimal system prompt, single-line instruction, no few-shot.
v2: full taxonomy with disambiguation notes, targeted few-shot examples for
    credential_harvesting and fear_intimidation (T1.2 spike found these
    systematically underdetected), explicit benign-only rule.
"""
from __future__ import annotations

from .types import TACTIC_TAXONOMY

_TAXONOMY_LINE = ", ".join(TACTIC_TAXONOMY)

_SYSTEM_V1 = (
    "You are a social-engineering analyst. Given a phone-call transcript, "
    "identify the persuasion tactics in use. Output a JSON array and nothing "
    "else. Each element: {label, confidence (0-1), evidenceSpans (verbatim quotes)}. "
    f"Valid labels: {_TAXONOMY_LINE}."
)

_SYSTEM_V2 = (
    "You are a social-engineering analyst. Classify persuasion tactics in a "
    "phone-call transcript. Output ONLY a JSON array — no prose before or after.\n\n"
    "Each element: {\"label\": <string>, \"confidence\": <float 0-1>, "
    "\"evidenceSpans\": [<verbatim quote>, ...]}\n\n"
    "Label definitions and disambiguation:\n"
    "  authority           — impersonates IRS, FBI, bank, government agency\n"
    "  fear_intimidation   — explicit threat: arrest, lawsuit, account closure, "
    "harm (use this when caller threatens consequences, not just creates urgency)\n"
    "  urgency             — time pressure without an explicit threat (\"act now\", "
    "\"today only\", \"expires soon\")\n"
    "  pretexting          — fabricated backstory that creates the scenario; "
    "almost always co-occurs with another tactic — do NOT use as a catch-all\n"
    "  credential_harvesting — asks for OTP, password, card number, SSN, "
    "PIN, or verification code (use even if phrased as 'confirm' or 'verify')\n"
    "  impersonation       — claims to be a known person (grandchild, colleague, "
    "friend), distinct from authority (government/institution)\n"
    "  financial_manipulation — requests wire transfer, gift cards, crypto, "
    "cash; or reverses a charge to extract payment\n"
    "  reward_prize        — claims caller won something; lottery, refund, gift card\n"
    "  tech_support        — claims device has virus/malware, requests remote access\n"
    "  benign              — use ONLY when NO attack tactics are present; "
    "do NOT add benign alongside malicious labels\n\n"
    f"All labels must be from: {_TAXONOMY_LINE}"
)

_FEW_SHOT_V2 = (
    "Example 1 — credential_harvesting:\n"
    "Transcript: \"This is your bank security team. We detected a login from "
    "Texas. Please read me the six-digit code we just sent to verify it was you.\"\n"
    "JSON: [{\"label\": \"authority\", \"confidence\": 0.9, "
    "\"evidenceSpans\": [\"your bank security team\"]}, "
    "{\"label\": \"credential_harvesting\", \"confidence\": 0.95, "
    "\"evidenceSpans\": [\"read me the six-digit code\"]}]\n\n"
    "Example 2 — fear_intimidation vs urgency:\n"
    "Transcript: \"This is the IRS. A warrant has been issued for your arrest. "
    "Pay $2400 in gift cards immediately or officers will be dispatched.\"\n"
    "JSON: [{\"label\": \"authority\", \"confidence\": 0.95, "
    "\"evidenceSpans\": [\"This is the IRS\"]}, "
    "{\"label\": \"fear_intimidation\", \"confidence\": 0.95, "
    "\"evidenceSpans\": [\"warrant has been issued for your arrest\", "
    "\"officers will be dispatched\"]}, "
    "{\"label\": \"financial_manipulation\", \"confidence\": 0.9, "
    "\"evidenceSpans\": [\"Pay $2400 in gift cards\"]}]\n\n"
    "Example 3 — benign:\n"
    "Transcript: \"Hey, it is Mike. Just calling to confirm dinner Friday at seven.\"\n"
    "JSON: [{\"label\": \"benign\", \"confidence\": 0.99, "
    "\"evidenceSpans\": []}]\n"
)


def tacticPromptV1(transcript: str) -> list[dict]:
    """Minimal prompt — bare instruction, no few-shot. Used as Phase 3 baseline."""
    return [
        {"role": "system", "content": _SYSTEM_V1},
        {"role": "user", "content": f"Transcript:\n{transcript}\n\nJSON:"},
    ]


def tacticPromptV2(transcript: str) -> list[dict]:
    """Full prompt — disambiguation notes + targeted few-shot. Phase 3 candidate."""
    return [
        {"role": "system", "content": _SYSTEM_V2 + "\n\n" + _FEW_SHOT_V2},
        {"role": "user", "content": f"Transcript:\n{transcript}\n\nJSON:"},
    ]


def riskReasoningPrompt(transcript: str, pSynthetic: float, tacticsJson: str) -> list[dict]:
    """Prompt for LLM-written risk reasoning string inside RiskScore."""
    return [
        {
            "role": "system",
            "content": (
                "You are a security analyst. Write a 2-4 sentence plain-English risk "
                "assessment for a phone call. Be concise and factual. "
                "Output only the assessment text — no JSON, no bullet points."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Synthetic voice probability: {pSynthetic:.2f}\n"
                f"Detected tactics: {tacticsJson}\n"
                f"Transcript excerpt: {transcript[:500]}\n\n"
                "Summarize the risk in 2-4 sentences:"
            ),
        },
    ]
