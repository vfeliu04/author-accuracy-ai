"""Helpers for interacting with OpenAI models with graceful fallbacks."""

from __future__ import annotations

import json
import re
from typing import Optional, Tuple

from openai import OpenAI

from .config import settings


class OpenAIUnavailable(RuntimeError):
    """Raised when OpenAI interactions are requested but unavailable."""


def ensure_openai_client() -> OpenAI:
    if not settings.openai_api_key:
        raise OpenAIUnavailable("OpenAI API key is not configured.")
    return OpenAI(api_key=settings.openai_api_key or None)


def generate_explanation(payload: dict) -> str:
    """Generate a concise explanation for a verdict."""
    client = ensure_openai_client()
    system_prompt = (
        "You are a precise scientific verifier. Use only the supplied claim and evidence text. "
        "Write one short paragraph that: (a) restates the claim, (b) cites the matching numbers/years from the evidence, "
        "and (c) states whether the evidence supports or contradicts the claim. Do not invent context or interpret the confidence value."
    )
    text = _run_model(
        client,
        model=settings.openai_chat_model,
        system_prompt=system_prompt,
        user_payload=payload,
        max_tokens=280,
    )
    if not text:
        raise RuntimeError("Failed to generate explanation from OpenAI.")
    return text.strip()


def score_relevance(claim_sentence: str, evidence_snippet: str, retrieval_score: float | None = None) -> Tuple[float, Optional[str]]:
    """Return (score, label) measuring how well the evidence supports the claim."""
    if not settings.rerank_with_gpt:
        raise OpenAIUnavailable("GPT reranking has been disabled via configuration.")

    client = ensure_openai_client()
    system_prompt = (
        "You evaluate how well evidence supports a claim. Respond with JSON "
        'like {"score": 0.82, "label": "supported"} where score is between 0 and 1. '
        "Use labels supported, contradicted, or irrelevant."
    )
    payload = {
        "claim": claim_sentence,
        "evidence": evidence_snippet,
        "retrieval_score": retrieval_score,
    }
    text = _run_model(
        client,
        model=settings.rerank_model,
        system_prompt=system_prompt,
        user_payload=payload,
        max_tokens=200,
    )
    if not text:
        raise RuntimeError("Failed to obtain rerank score from OpenAI.")
    return _parse_score_response(text)


def confirm_entity_alignment(claim_sentence: str, evidence_snippet: str) -> Tuple[Optional[bool], Optional[float], Optional[str]]:
    """Use GPT to judge whether claim and evidence describe the same entities."""
    client = ensure_openai_client()
    system_prompt = (
        "Determine whether the claim text and the evidence refer to the same primary entities. "
        "Return JSON like {\"match\": true, \"confidence\": 0.82, \"reason\": \"entities align\"}. "
        "Use confidence between 0 and 1 to reflect how certain you are. "
        "Answer true only when the subjects and objects truly match; if the evidence talks about different items, respond with false."
    )
    payload = {
        "claim": claim_sentence,
        "evidence": evidence_snippet,
    }
    text = _run_model(
        client,
        model=settings.rerank_model,
        system_prompt=system_prompt,
        user_payload=payload,
        max_tokens=180,
    )
    if not text:
        return None, None, None
    match, confidence, reason = _parse_alignment_response(text)
    return match, confidence, reason


def _run_model(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    user_payload: dict,
    max_tokens: int,
) -> Optional[str]:
    text = _call_responses_api(
        client,
        model=model,
        system_prompt=system_prompt,
        user_payload=user_payload,
        max_tokens=max_tokens,
    )
    if text:
        return text

    fallback_model = model
    if fallback_model.startswith("gpt-4.1"):
        fallback_model = "gpt-4o-mini"

    text = _call_chat_completions(
        client,
        model=fallback_model,
        system_prompt=system_prompt,
        user_payload=user_payload,
        max_tokens=max_tokens,
    )
    return text


def _call_responses_api(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    user_payload: dict,
    max_tokens: int,
) -> Optional[str]:
    if not hasattr(client, "responses"):
        return None
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        temperature=0,
        max_output_tokens=max_tokens,
    )
    return _extract_responses_text(response)


def _call_chat_completions(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    user_payload: dict,
    max_tokens: int,
) -> Optional[str]:
    if not hasattr(client, "chat") or not hasattr(client.chat, "completions"):
        return None
    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    )
    choice = (completion.choices or [None])[0]
    if not choice or not getattr(choice, "message", None):
        return None
    content = getattr(choice.message, "content", None)
    return content.strip() if content else None


def _extract_responses_text(response) -> Optional[str]:
    text = getattr(response, "output_text", None)
    if text:
        return text.strip()

    output = getattr(response, "output", None)
    if not output:
        return None
    fragments: list[str] = []
    try:
        for item in output:
            for part in getattr(item, "content", []):
                fragment = getattr(part, "text", None)
                if fragment:
                    fragments.append(fragment)
    except Exception:
        return None

    joined = "".join(fragments).strip()
    return joined or None


def _parse_score_response(text: str) -> Tuple[float, Optional[str]]:
    try:
        data = json.loads(text)
        score = float(data.get("score"))
        label = data.get("label")
        return _clamp_score(score), _normalize_label(label)
    except Exception:
        pass

    number_match = re.search(r"0?\.\d+|1(\.0+)?", text)
    score = float(number_match.group(0)) if number_match else 0.0
    label_match = re.search(r"(supported|contradicted|irrelevant)", text, re.IGNORECASE)
    label = label_match.group(1).lower() if label_match else None
    return _clamp_score(score), _normalize_label(label)


def _parse_alignment_response(text: str) -> Tuple[Optional[bool], Optional[float], Optional[str]]:
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            match_value = data.get("match")
            confidence_value = data.get("confidence") or data.get("certainty") or data.get("score")
            reason = data.get("reason") or data.get("explanation")
            confidence = _safe_float(confidence_value)
            if isinstance(match_value, bool):
                return match_value, confidence, reason
            if isinstance(match_value, str):
                lowered = match_value.strip().lower()
                if lowered in {"true", "yes", "match", "matched", "aligned", "same"}:
                    return True, confidence, reason
                if lowered in {"false", "no", "different", "mismatch", "not"}:
                    return False, confidence, reason
        if isinstance(data, bool):
            return data, None, None
    except Exception:
        pass

    lowered_text = text.lower()
    if "\"match\"" in lowered_text:
        bool_match = re.search(r"\"match\"\s*:\s*(true|false)", lowered_text)
        if bool_match:
            inferred_conf = None
            confidence_match = re.search(r"\"confidence\"\s*:\s*(0?\.\d+|1(?:\.0+)?)", lowered_text)
            if confidence_match:
                inferred_conf = _safe_float(confidence_match.group(1))
            return bool_match.group(1) == "true", inferred_conf, None
    if "true" in lowered_text and "false" not in lowered_text:
        return True, None, None
    if "false" in lowered_text and "true" not in lowered_text:
        return False, None, None
    return None, None, None


def _clamp_score(value: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _normalize_label(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    lowered = label.lower()
    if lowered in {"supported", "support"}:
        return "supported"
    if lowered in {"contradicted", "conflict"}:
        return "contradicted"
    if lowered in {"irrelevant", "not_found", "not found"}:
        return "irrelevant"
    return lowered


def _safe_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None
