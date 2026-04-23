"""Open-LLM tactic classification stage — Transcript -> tuple[Tactic, ...]."""
from __future__ import annotations

import json
import re

from .promptLibrary import tacticPromptV1, tacticPromptV2, tacticPromptV3, tacticPromptV4
from .types import TACTIC_TAXONOMY, LlmConfig, Tactic, Transcript

_MODEL_CACHE: dict = {}  # model_id -> (tokenizer, model)
_TAXONOMY_SET: frozenset[str] = frozenset(TACTIC_TAXONOMY)


def _load_tokenizer_and_model(cfg: LlmConfig):
    if cfg.modelId not in _MODEL_CACHE:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.modelId)

        if cfg.loadIn4Bit:
            from transformers import BitsAndBytesConfig

            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            )
            model = AutoModelForCausalLM.from_pretrained(
                cfg.modelId,
                quantization_config=quant_cfg,
                device_map="auto",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                cfg.modelId,
                torch_dtype=torch.float16,
                device_map="auto" if cfg.device == "cuda" else "cpu",
            )

        model.eval()
        _MODEL_CACHE[cfg.modelId] = (tokenizer, model)
    return _MODEL_CACHE[cfg.modelId]


def _call_llm(messages: list[dict], cfg: LlmConfig) -> str:
    import torch

    tokenizer, model = _load_tokenizer_and_model(cfg)
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=cfg.maxNewTokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(
        outputs[0][len(inputs["input_ids"][0]):],
        skip_special_tokens=True,
    ).strip()


def _extract_json(text: str) -> list[dict]:
    """Extract first JSON array from model output, raising ValueError on failure.

    Uses bracket counting so the regex doesn't overshoot the closing ] when
    the model emits prose after the array. Also normalises Python-style single
    quotes to double quotes as a fallback before giving up.
    """
    start = text.find("[")
    if start == -1:
        raise ValueError(f"No JSON array found in LLM output: {text[:200]!r}")

    depth = 0
    in_str = False
    escape_next = False
    end = -1
    for i, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_str:
            escape_next = True
            continue
        if ch == '"':
            in_str = not in_str
        if in_str:
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end == -1:
        raise ValueError(f"Unmatched '[' in LLM output: {text[:200]!r}")

    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Fallback: replace Python-style single-quoted strings with double quotes.
    normalised = re.sub(r"'([^']*)'", r'"\1"', candidate)
    try:
        return json.loads(normalised)
    except json.JSONDecodeError:
        raise ValueError(f"JSON parse failed even after quote normalisation: {candidate[:200]!r}")


def _parse_tactics(raw: list) -> tuple[Tactic, ...]:
    """Filter to valid taxonomy labels and coerce field types.

    Accepts two formats the model may return:
      - list of dicts: [{"label": "urgency", "confidence": 0.9, "evidenceSpans": [...]}]
      - flat string list: ["urgency", "authority"]
    """
    result = []
    for item in raw:
        if isinstance(item, str):
            label = item.strip()
            if label in _TAXONOMY_SET:
                result.append(Tactic(label=label, confidence=0.5, evidenceSpans=()))
        elif isinstance(item, dict):
            label = str(item.get("label", "")).strip()
            if label not in _TAXONOMY_SET:
                continue
            try:
                confidence = float(item.get("confidence", 0.5))
            except (ValueError, TypeError):
                confidence = 0.5
            spans = tuple(str(s) for s in item.get("evidenceSpans", []))
            result.append(Tactic(label=label, confidence=confidence, evidenceSpans=spans))
    return tuple(result)


def classifyTactics(transcript: Transcript, cfg: LlmConfig) -> tuple[Tactic, ...]:
    if cfg.promptVariant == "v4":
        prompt_fn = tacticPromptV4
    elif cfg.promptVariant == "v3":
        prompt_fn = tacticPromptV3
    elif cfg.promptVariant == "v2":
        prompt_fn = tacticPromptV2
    else:
        prompt_fn = tacticPromptV1
    messages = prompt_fn(transcript.fullText)

    raw_output = _call_llm(messages, cfg)

    try:
        raw_list = _extract_json(raw_output)
    except (ValueError, json.JSONDecodeError):
        retry_messages = [
            {"role": "system", "content": "Output ONLY a valid JSON array. No prose."},
            *messages[1:],
            {"role": "assistant", "content": raw_output},
            {"role": "user", "content": "Try again. Output ONLY the JSON array:"},
        ]
        raw_output2 = _call_llm(retry_messages, cfg)
        try:
            raw_list = _extract_json(raw_output2)
        except (ValueError, json.JSONDecodeError):
            return ()  # both attempts failed — count as no tactics detected

    return _parse_tactics(raw_list)
