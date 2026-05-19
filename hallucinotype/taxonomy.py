"""
HallucinoType Taxonomy
======================
Defines the hallucination type enum, severity levels, evidence model,
and the HallucinationFingerprint output schema.

Paper reference: "HallucinoType: A Taxonomy and Detection Framework
for LLM Hallucination Patterns"
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Hallucination Type Taxonomy
# ---------------------------------------------------------------------------

class HallucinationType(str, Enum):
    """
    Eight canonical hallucination types.

    Each type represents a distinct failure mode with different causes
    and different mitigation strategies.
    """

    # Wrong entity used in place of the correct one.
    # Example: attributing a quote by Einstein to Bohr.
    ENTITY_SUBSTITUTION = "entity_substitution"

    # Incorrect date, era, or temporal ordering.
    # Example: claiming the Berlin Wall fell in 1991 instead of 1989.
    TEMPORAL_CONFUSION = "temporal_confusion"

    # Facts from multiple distinct sources merged into one incorrect claim.
    # Example: combining details from two different studies into a single result.
    SOURCE_BLENDING = "source_blending"

    # Fully fabricated claim with no factual basis, stated confidently.
    # The "hallucination" in the classic sense.
    CONFIDENT_FABRICATION = "confident_fabrication"

    # Correct entities and relations but wrong numbers or statistics.
    # Example: citing a study's 23% finding as 53%.
    NUMERICAL_DISTORTION = "numerical_distortion"

    # Correct entities, wrong relationship between them.
    # Example: "X acquired Y" when Y acquired X.
    RELATION_ERROR = "relation_error"

    # Logical polarity inverted — claim asserts opposite of truth.
    # Example: "the vaccine did not show efficacy" for a trial that did.
    NEGATION_FLIP = "negation_flip"

    # Specific fact incorrectly generalized to a broader claim.
    # Example: one study's result stated as universal scientific consensus.
    OVERGENERALIZATION = "overgeneralization"


class HallucinationSeverity(str, Enum):
    """
    How badly does this hallucination mislead a reader?

    Severity is independent of type: any type can occur at any severity.
    """
    LOW = "low"         # Minor inaccuracy, unlikely to cause real harm
    MEDIUM = "medium"   # Meaningful error, could mislead a careful reader
    HIGH = "high"       # Seriously misleading, likely to propagate


# ---------------------------------------------------------------------------
# Evidence model
# ---------------------------------------------------------------------------

class Evidence(BaseModel):
    """
    A single piece of evidence supporting a hallucination detection.

    Detectors attach evidence to explain *why* they flagged something,
    making outputs auditable and useful for downstream analysis.
    """
    hallucination_type: HallucinationType
    source: str
    description: str
    span: Optional[tuple[int, int]] = None
    reference_text: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Core fingerprint output
# ---------------------------------------------------------------------------

class HallucinationFingerprint(BaseModel):
    """
    The main output of the HallucinoType detection pipeline.

    One fingerprint per (claim, context) pair. A claim can carry
    multiple hallucination types simultaneously.
    """
    claim: str
    context: Optional[str] = None
    detected_types: dict[HallucinationType, float] = Field(default_factory=dict)
    severity: dict[HallucinationType, HallucinationSeverity] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)
    hallucination_probability: float = 0.0
    dominant_type: Optional[HallucinationType] = None
    judge_response: Optional[str] = None

    def is_hallucinated(self, threshold: float = 0.5) -> bool:
        """True if any detected type exceeds the confidence threshold."""
        return any(conf >= threshold for conf in self.detected_types.values())

    def top_types(self, n: int = 3) -> list[tuple[HallucinationType, float]]:
        """Return top-n detected types sorted by confidence."""
        return sorted(
            self.detected_types.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]

    def summary(self) -> str:
        """Human-readable one-line summary."""
        if not self.detected_types:
            return "No hallucination detected."
        types_str = ", ".join(
            f"{t.value} ({c:.2f})"
            for t, c in self.top_types()
        )
        return (
            f"Hallucination detected [p={self.hallucination_probability:.2f}]: {types_str}"
        )

    def to_dict(self) -> dict:
        d = self.model_dump(mode="json")
        d["is_hallucinated"] = self.is_hallucinated()
        return d
