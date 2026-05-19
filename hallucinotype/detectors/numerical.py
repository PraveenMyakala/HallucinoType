"""
Numerical Distortion Detector
===============================
Detects claims where numbers or statistics are wrong while the
surrounding context (entities, relations) is correct.

Strategy:
  1. Extract all numeric expressions from claim and context.
  2. Match numerics by proximity of surrounding context words.
  3. Flag large relative differences as distortion.
  4. Handles percentages, counts, monetary values, and ratios.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from hallucinotype.detectors.base import BaseDetector
from hallucinotype.taxonomy import Evidence, HallucinationType

# ---------------------------------------------------------------------------
# Numeric extraction
# ---------------------------------------------------------------------------

@dataclass
class NumericSpan:
    raw: str            # Original text
    value: float        # Parsed numeric value
    unit: str           # "%", "$", "million", "billion", "", etc.
    start: int
    end: int
    window: str         # 10 words of surrounding context


# Matches: 3.5%, $1.2 billion, 47, 2,400, 1.5M, etc.
NUMBER_RE = re.compile(
    r'(?P<currency>\$|€|£)?'
    r'(?P<number>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)'
    r'\s*(?P<suffix>million|billion|trillion|thousand|M|B|K|%)?',
    re.IGNORECASE
)

SUFFIX_MULTIPLIERS = {
    "million": 1e6, "M": 1e6,
    "billion": 1e9, "B": 1e9,
    "trillion": 1e12,
    "thousand": 1e3, "K": 1e3,
}


def _parse_number(raw: str, number: str, suffix: str) -> float:
    val = float(number.replace(",", ""))
    multiplier = SUFFIX_MULTIPLIERS.get(suffix, 1.0) if suffix else 1.0
    return val * multiplier


def _context_window(text: str, start: int, end: int, words: int = 8) -> str:
    """Extract N words before and after a span."""
    # Extract slightly more text, discard the potentially truncated boundary token
    before_tokens = text[max(0, start - 80):start].split()
    before = before_tokens[1:][-words:] if len(before_tokens) > 1 else before_tokens[-words:]
    after_tokens = text[end:end + 80].split()
    after = after_tokens[:-1][:words] if len(after_tokens) > 1 else after_tokens[:words]
    return " ".join(before + after).lower()


def extract_numerics(text: str) -> list[NumericSpan]:
    spans = []
    for m in NUMBER_RE.finditer(text):
        if not m.group("number"):
            continue
        raw = m.group()
        try:
            val = _parse_number(
                raw,
                m.group("number"),
                m.group("suffix") or ""
            )
        except ValueError:
            continue

        unit = ""
        if m.group("currency"):
            unit = m.group("currency")
        if m.group("suffix"):
            unit += m.group("suffix")

        spans.append(NumericSpan(
            raw=raw,
            value=val,
            unit=unit,
            start=m.start(),
            end=m.end(),
            window=_context_window(text, m.start(), m.end()),
        ))
    return spans


def _window_overlap_score(w1: str, w2: str) -> float:
    """
    Bag-of-words overlap between two context windows.
    Returns [0, 1].
    """
    s1 = set(w1.split())
    s2 = set(w2.split())
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def _relative_error(claim_val: float, context_val: float) -> float:
    """Relative error: |claim - context| / max(|context|, 1)."""
    return abs(claim_val - context_val) / max(abs(context_val), 1.0)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class NumericalDistortionDetector(BaseDetector):
    """
    Detects wrong numbers in claims by comparing against context numerics.

    Two numbers are "paired" when their surrounding words overlap enough
    to suggest they're referring to the same quantity. A large relative
    difference between paired numbers is flagged as distortion.
    """

    detects = HallucinationType.NUMERICAL_DISTORTION

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        window_overlap_threshold: float = 0.3,
        relative_error_threshold: float = 0.15,
    ):
        super().__init__(confidence_threshold)
        self.window_overlap_threshold = window_overlap_threshold
        self.relative_error_threshold = relative_error_threshold

    def detect(
        self,
        claim: str,
        context: Optional[str] = None,
    ) -> list[Evidence]:
        if context is None:
            return []

        claim_nums = extract_numerics(claim)
        context_nums = extract_numerics(context)

        if not claim_nums or not context_nums:
            return []

        evidence = []

        for cn in claim_nums:
            # Find the best-matching context number by window overlap
            best_match: Optional[NumericSpan] = None
            best_overlap = 0.0

            for ctx_n in context_nums:
                overlap = _window_overlap_score(cn.window, ctx_n.window)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = ctx_n

            if best_match is None or best_overlap < self.window_overlap_threshold:
                continue  # Couldn't confidently pair this number

            rel_err = _relative_error(cn.value, best_match.value)

            if rel_err > self.relative_error_threshold:
                # Scale confidence with relative error magnitude
                confidence = min(0.95, 0.4 + rel_err * 0.5)

                if confidence >= self.confidence_threshold:
                    evidence.append(Evidence(
                        source="NumericalDistortionDetector",
                        description=(
                            f"Claim uses '{cn.raw}' (~{cn.value:,.0f}) but context "
                            f"has '{best_match.raw}' (~{best_match.value:,.0f}). "
                            f"Relative error: {rel_err:.1%}. "
                            f"Context window overlap: {best_overlap:.2f}."
                        ),
                        span=(cn.start, cn.end),
                        reference_text=best_match.raw,
                        confidence=confidence,
                    ))

        return evidence
