"""
Temporal Confusion Detector
============================
Detects incorrect dates, years, eras, or temporal ordering in claims.

Strategy:
  1. Extract all date/year expressions from claim and context using
     regex + dateparser.
  2. Compare claim dates against context dates.
  3. Flag mismatches above a configurable tolerance.
  4. Also detects era-level confusion (e.g., claiming a 19th-century
     event happened in the 20th century).
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from hallucinotype.detectors.base import BaseDetector
from hallucinotype.taxonomy import Evidence, HallucinationType

# ---------------------------------------------------------------------------
# Year extraction helpers
# ---------------------------------------------------------------------------

# Matches 4-digit years in plausible historical range
YEAR_PATTERN = re.compile(r'\b(1[0-9]{3}|20[0-2][0-9])\b')

# Decade references: "the 1980s", "the 90s", "the eighties"
DECADE_PATTERN = re.compile(
    r'\b(?:the\s+)?'
    r'((?:19|20)\d0s'
    r'|(?:twenties|thirties|forties|fifties|sixties|seventies|eighties|nineties))'
    r'\b',
    re.IGNORECASE
)

DECADE_WORD_MAP = {
    "twenties": 1920, "thirties": 1930, "forties": 1940,
    "fifties": 1950, "sixties": 1960, "seventies": 1970,
    "eighties": 1980, "nineties": 1990,
}

# Month + year patterns: "March 1989", "in 1989", "January of 2001"
MONTH_YEAR_PATTERN = re.compile(
    r'\b(?:January|February|March|April|May|June|July|August|'
    r'September|October|November|December)\s+(?:of\s+)?(\d{4})\b',
    re.IGNORECASE
)


def extract_years(text: str) -> list[tuple[int, int, int]]:
    """
    Extract (year, start_char, end_char) from text.
    Returns list sorted by position.
    """
    results = []
    for m in YEAR_PATTERN.finditer(text):
        results.append((int(m.group()), m.start(), m.end()))
    for m in DECADE_PATTERN.finditer(text):
        decade_str = m.group(1).lower()
        if decade_str.endswith("s"):
            try:
                year = int(decade_str[:4])
                results.append((year, m.start(), m.end()))
            except ValueError:
                pass
        elif decade_str in DECADE_WORD_MAP:
            results.append((DECADE_WORD_MAP[decade_str], m.start(), m.end()))
    results.sort(key=lambda x: x[1])
    return results


def extract_full_dates(text: str) -> list[tuple[datetime, int, int]]:
    """
    Extract full date objects where month+year is specified.
    Returns list of (datetime, start, end).
    """
    results = []
    for m in MONTH_YEAR_PATTERN.finditer(text):
        try:
            dt = datetime.strptime(m.group(), "%B %Y")
            results.append((dt, m.start(), m.end()))
        except ValueError:
            try:
                dt = datetime.strptime(m.group(), "%B of %Y")
                results.append((dt, m.start(), m.end()))
            except ValueError:
                pass
    return results


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class TemporalConfusionDetector(BaseDetector):
    """
    Detects temporal hallucinations: wrong years, dates, or eras.

    Year tolerance: how many years off counts as an error.
    Default is 0 (exact match required). For historical claims
    where approximate dating is acceptable, set tolerance to 5 or 10.
    """

    detects = HallucinationType.TEMPORAL_CONFUSION

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        year_tolerance: int = 0,
    ):
        super().__init__(confidence_threshold)
        self.year_tolerance = year_tolerance

    def detect(
        self,
        claim: str,
        context: Optional[str] = None,
    ) -> list[Evidence]:
        if context is None:
            return []

        claim_years = extract_years(claim)
        context_years = extract_years(context)

        if not claim_years or not context_years:
            return []

        context_year_set = {y for y, _, _ in context_years}
        evidence = []

        for year, start, end in claim_years:
            # Check exact match first
            if year in context_year_set:
                continue

            # Check within tolerance
            nearest_context_year = min(
                context_year_set,
                key=lambda cy: abs(cy - year)
            )
            gap = abs(nearest_context_year - year)

            if gap > self.year_tolerance:
                # Scale confidence with how far off the year is
                # A 1-year error is different from a 10-year error
                confidence = min(0.95, 0.4 + (gap / 50))
                evidence.append(Evidence(
                    source="TemporalConfusionDetector",
                    description=(
                        f"Year {year} in claim doesn't match context. "
                        f"Nearest year in context: {nearest_context_year} "
                        f"(gap: {gap} years)."
                    ),
                    span=(start, end),
                    reference_text=str(nearest_context_year),
                    confidence=confidence,
                ))

        # Check full date mismatches (month + year level)
        claim_dates = extract_full_dates(claim)
        context_dates = extract_full_dates(context)

        if claim_dates and context_dates:
            context_date_set = {dt for dt, _, _ in context_dates}
            for dt, start, end in claim_dates:
                exact_matches = [
                    cd for cd in context_date_set
                    if cd.year == dt.year and cd.month == dt.month
                ]
                if not exact_matches:
                    nearest = min(
                        context_date_set,
                        key=lambda cd: abs((cd - dt).days)
                    )
                    gap_days = abs((nearest - dt).days)
                    if gap_days > 31:  # More than a month off
                        confidence = min(0.95, 0.5 + gap_days / 365)
                        evidence.append(Evidence(
                            source="TemporalConfusionDetector",
                            description=(
                                f"Date '{dt.strftime('%B %Y')}' in claim doesn't "
                                f"match context (nearest: {nearest.strftime('%B %Y')}, "
                                f"gap: {gap_days} days)."
                            ),
                            span=(start, end),
                            reference_text=nearest.strftime("%B %Y"),
                            confidence=confidence,
                        ))

        return evidence
