"""
LLM Judge Detector
==================
Uses an LLM to detect hallucination types that are hard to catch
with rule-based methods:

  - CONFIDENT_FABRICATION
  - SOURCE_BLENDING
  - RELATION_ERROR
  - NEGATION_FLIP
  - OVERGENERALIZATION

The judge is prompted with a structured template that asks it to
reason step-by-step, output a JSON verdict, and cite specific spans.

Supports both Anthropic (Claude) and OpenAI (GPT-4) backends.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

from hallucinotype.detectors.base import BaseDetector
from hallucinotype.taxonomy import Evidence, HallucinationType


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a hallucination detection expert. Your job is to
analyze a CLAIM against a REFERENCE CONTEXT and identify specific hallucination types.

You must respond with a valid JSON object only. No prose before or after the JSON.

The JSON schema is:
{
  "detected": [
    {
      "type": "<hallucination_type>",
      "confidence": <float 0.0 to 1.0>,
      "span": "<exact substring from the claim that is hallucinated>",
      "reference_text": "<correct value from context, or null>",
      "explanation": "<one sentence explanation>"
    }
  ],
  "reasoning": "<brief chain of thought before reaching verdict>"
}

Hallucination types you can detect:
- confident_fabrication: Claim states something with no basis in context or common knowledge
- source_blending: Claim mixes facts from different things/events/people into one incorrect statement
- relation_error: Correct entities mentioned but the relationship between them is wrong (e.g., reversed acquisition)
- negation_flip: Claim asserts the logical opposite of what is true (e.g., says X did NOT happen when it did)
- overgeneralization: Claim takes a specific fact and incorrectly applies it universally

If no hallucination is detected, return: {"detected": [], "reasoning": "<your reasoning>"}

Be precise about spans — quote the exact words from the claim that are wrong.
Only flag something if you are at least 40% confident."""

USER_TEMPLATE = """CLAIM:
{claim}

REFERENCE CONTEXT:
{context}

Analyze the claim for hallucinations against the reference context."""

USER_TEMPLATE_NO_CONTEXT = """CLAIM:
{claim}

No reference context was provided. Analyze the claim for hallucinations
based on your knowledge of established facts. Only flag confident_fabrication
or negation_flip when you are highly certain something is factually wrong."""


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------

def _call_anthropic(prompt_messages: list[dict], model: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=prompt_messages,
    )
    return response.content[0].text


def _call_openai(prompt_messages: list[dict], model: str) -> str:
    import openai
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + prompt_messages
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1024,
        temperature=0.0,
    )
    return response.choices[0].message.content


def _parse_judge_response(raw: str, claim: str) -> tuple[list[Evidence], str]:
    """
    Parse the judge's JSON response into Evidence objects.
    Returns (evidence_list, raw_response).
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: try to extract JSON object
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return [], raw
        else:
            return [], raw

    evidence = []
    for det in data.get("detected", []):
        try:
            h_type_str = det.get("type", "")
            # Validate that it's a known type
            try:
                HallucinationType(h_type_str)
            except ValueError:
                continue

            span_text = det.get("span", "")
            start, end = None, None
            if span_text:
                idx = claim.find(span_text)
                if idx != -1:
                    start, end = idx, idx + len(span_text)

            evidence.append(Evidence(
                source="LLMJudgeDetector",
                description=det.get("explanation", ""),
                span=(start, end) if start is not None else None,
                reference_text=det.get("reference_text"),
                confidence=float(det.get("confidence", 0.5)),
            ))
        except (KeyError, TypeError, ValueError):
            continue

    reasoning = data.get("reasoning", "")
    return evidence, reasoning


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class LLMJudgeDetector(BaseDetector):
    """
    LLM-as-judge detector for semantically complex hallucination types.

    Handles: confident_fabrication, source_blending, relation_error,
    negation_flip, overgeneralization.

    Use alongside rule-based detectors (entity, temporal, numerical)
    for full coverage. The judge is slower and costs API tokens, so
    it's designed to catch what rules can't.
    """

    # This detector handles multiple types
    detects = HallucinationType.CONFIDENT_FABRICATION  # Primary type

    DETECTABLE_TYPES = {
        HallucinationType.CONFIDENT_FABRICATION,
        HallucinationType.SOURCE_BLENDING,
        HallucinationType.RELATION_ERROR,
        HallucinationType.NEGATION_FLIP,
        HallucinationType.OVERGENERALIZATION,
    }

    def __init__(
        self,
        backend: str = "anthropic",
        model: Optional[str] = None,
        confidence_threshold: float = 0.4,
    ):
        super().__init__(confidence_threshold)
        self.backend = backend

        if model is None:
            self.model = (
                "claude-haiku-4-5-20251001" if backend == "anthropic"
                else "gpt-4o-mini"
            )
        else:
            self.model = model

        self._raw_response: Optional[str] = None

    def detect(
        self,
        claim: str,
        context: Optional[str] = None,
    ) -> list[Evidence]:
        if context:
            user_content = USER_TEMPLATE.format(claim=claim, context=context)
        else:
            user_content = USER_TEMPLATE_NO_CONTEXT.format(claim=claim)

        messages = [{"role": "user", "content": user_content}]

        try:
            if self.backend == "anthropic":
                raw = _call_anthropic(messages, self.model)
            elif self.backend == "openai":
                raw = _call_openai(messages, self.model)
            else:
                raise ValueError(f"Unknown backend: {self.backend}")
        except Exception as e:
            # Don't crash the pipeline if the API call fails
            self._raw_response = f"API error: {e}"
            return []

        self._raw_response = raw
        evidence, _ = _parse_judge_response(raw, claim)

        # Filter by confidence threshold
        return [e for e in evidence if e.confidence >= self.confidence_threshold]

    def detect_with_reasoning(
        self,
        claim: str,
        context: Optional[str] = None,
    ) -> tuple[list[Evidence], str]:
        """
        Same as detect() but also returns the judge's chain-of-thought.
        Useful for debugging and auditing.
        """
        self.detect(claim, context)
        if self._raw_response:
            cleaned = re.sub(r"```(?:json)?", "", self._raw_response).strip()
            try:
                data = json.loads(cleaned)
                reasoning = data.get("reasoning", "")
            except json.JSONDecodeError:
                reasoning = self._raw_response
        else:
            reasoning = ""

        evidence, _ = _parse_judge_response(self._raw_response or "", claim)
        return (
            [e for e in evidence if e.confidence >= self.confidence_threshold],
            reasoning
        )
