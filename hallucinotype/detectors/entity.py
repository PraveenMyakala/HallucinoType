"""
Entity Substitution Detector
=============================
Detects when a named entity in the claim (person, org, location, etc.)
doesn't match entities in the reference context.

Strategy:
  1. Extract named entities from claim and context using spaCy NER.
  2. For each entity in the claim, check if it appears in context.
  3. If a claim entity is absent from context but a same-type entity
     exists in context, flag as potential substitution.
  4. Confidence scales with: entity type importance, edit distance
     between claim entity and nearest context entity.
"""

from __future__ import annotations

import re
import threading
from typing import Optional

from hallucinotype.detectors.base import BaseDetector
from hallucinotype.taxonomy import Evidence, HallucinationType

# spaCy is loaded lazily so the import doesn't crash if it's not installed
_nlp = None
_nlp_lock = threading.Lock()

# Entity types ranked by substitution risk
# PERSON and ORG substitutions are most dangerous
ENTITY_TYPE_WEIGHTS = {
    "PERSON": 1.0,
    "ORG": 0.9,
    "GPE": 0.8,     # Geopolitical entity (countries, cities)
    "LOC": 0.7,
    "PRODUCT": 0.6,
    "EVENT": 0.6,
    "WORK_OF_ART": 0.5,
    "LAW": 0.5,
    "LANGUAGE": 0.4,
    "NORP": 0.4,    # Nationalities, religious groups
}


def _get_nlp():
    global _nlp
    if _nlp is None:
        with _nlp_lock:
            if _nlp is None:  # double-checked locking: only the first thread loads
                try:
                    import spacy
                    _nlp = spacy.load("en_core_web_sm")
                except OSError:
                    raise RuntimeError(
                        "spaCy model not found. Run: python -m spacy download en_core_web_sm"
                    )
    return _nlp


def _edit_distance(a: str, b: str) -> int:
    """Simple Levenshtein distance."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]


def _normalized_similarity(a: str, b: str) -> float:
    """Edit distance normalized to [0, 1]. 1.0 = identical."""
    if not a and not b:
        return 1.0
    max_len = max(len(a), len(b))
    return 1.0 - _edit_distance(a.lower(), b.lower()) / max_len


class EntitySubstitutionDetector(BaseDetector):
    """
    Detects entity substitution hallucinations using NER comparison.

    Works best when context is a reference document or retrieved passage.
    Falls back to a lightweight regex-based entity check when spaCy
    is unavailable.
    """

    detects = HallucinationType.ENTITY_SUBSTITUTION

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        similarity_cutoff: float = 0.75,
        use_spacy: bool = True,
    ):
        super().__init__(confidence_threshold)
        self.similarity_cutoff = similarity_cutoff
        self.use_spacy = use_spacy

    def detect(
        self,
        claim: str,
        context: Optional[str] = None,
    ) -> list[Evidence]:
        if context is None:
            # Without context we can't compare entities
            return []

        if self.use_spacy:
            return self._detect_with_spacy(claim, context)
        else:
            return self._detect_with_regex(claim, context)

    def _detect_with_spacy(self, claim: str, context: str) -> list[Evidence]:
        nlp = _get_nlp()
        claim_doc = nlp(claim)
        context_doc = nlp(context)

        claim_entities = [
            (ent.text, ent.label_, ent.start_char, ent.end_char)
            for ent in claim_doc.ents
            if ent.label_ in ENTITY_TYPE_WEIGHTS
        ]
        context_entity_texts = {
            ent.text.lower() for ent in context_doc.ents
        }

        evidence = []
        for ent_text, ent_type, start, end in claim_entities:
            # Exact match in context — no issue
            if ent_text.lower() in context_entity_texts:
                continue

            # Find closest context entity of same type
            same_type_context = [
                ent.text for ent in context_doc.ents
                if ent.label_ == ent_type
            ]

            if not same_type_context:
                # Entity type not in context at all — weaker signal
                confidence = ENTITY_TYPE_WEIGHTS.get(ent_type, 0.3) * 0.5
                evidence.append(Evidence(
                    hallucination_type=HallucinationType.ENTITY_SUBSTITUTION,
                    source="EntitySubstitutionDetector",
                    description=(
                        f"Entity '{ent_text}' ({ent_type}) appears in claim "
                        f"but no {ent_type} entities found in context."
                    ),
                    span=(start, end),
                    reference_text=None,
                    confidence=confidence,
                ))
                continue

            # Find nearest context entity by string similarity
            best_match = max(
                same_type_context,
                key=lambda x: _normalized_similarity(ent_text, x)
            )
            similarity = _normalized_similarity(ent_text, best_match)

            # If claim entity is close-but-not-exact to a context entity
            # that's our substitution signal
            if similarity < 1.0 and similarity > 0.3:
                type_weight = ENTITY_TYPE_WEIGHTS.get(ent_type, 0.3)
                # Higher confidence when similar but not identical
                confidence = type_weight * (1.0 - similarity)
                if confidence >= self.confidence_threshold:
                    evidence.append(Evidence(
                        hallucination_type=HallucinationType.ENTITY_SUBSTITUTION,
                        source="EntitySubstitutionDetector",
                        description=(
                            f"'{ent_text}' in claim closely resembles '{best_match}' "
                            f"in context (similarity={similarity:.2f}). "
                            f"Possible entity substitution."
                        ),
                        span=(start, end),
                        reference_text=best_match,
                        confidence=min(confidence, 0.95),
                    ))

        return evidence

    def _detect_with_regex(self, claim: str, context: str) -> list[Evidence]:
        """
        Lightweight fallback: extracts capitalized proper noun phrases
        and checks whether they appear in context.
        """
        # Find capitalized multi-word phrases (rough proper noun proxy)
        pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        claim_names = re.findall(pattern, claim)
        context_lower = context.lower()

        evidence = []
        for name in claim_names:
            if name.lower() not in context_lower and len(name) > 3:
                evidence.append(Evidence(
                    hallucination_type=HallucinationType.ENTITY_SUBSTITUTION,
                    source="EntitySubstitutionDetector(regex)",
                    description=(
                        f"Proper noun '{name}' in claim not found in context."
                    ),
                    span=None,
                    reference_text=None,
                    confidence=0.4,  # Low confidence without NER
                ))
        return evidence
