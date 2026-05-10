"""
HallucinoType Pipeline
=======================
Orchestrates all detectors and aggregates their outputs into
a single HallucinationFingerprint per (claim, context) pair.

Usage:
    from hallucinotype import HallucinoTypePipeline

    pipeline = HallucinoTypePipeline()
    result = pipeline.run(
        claim="Einstein won the Nobel Prize in 1905 for the theory of relativity.",
        context="Albert Einstein received the Nobel Prize in Physics in 1921 for his discovery of the photoelectric effect."
    )
    print(result.summary())
    # Hallucination detected [p=0.87]: temporal_confusion (0.85), entity_substitution (0.72)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from hallucinotype.detectors.base import BaseDetector
from hallucinotype.detectors.entity import EntitySubstitutionDetector
from hallucinotype.detectors.llm_judge import LLMJudgeDetector
from hallucinotype.detectors.numerical import NumericalDistortionDetector
from hallucinotype.detectors.temporal import TemporalConfusionDetector
from hallucinotype.taxonomy import (
    Evidence,
    HallucinationFingerprint,
    HallucinationSeverity,
    HallucinationType,
)


# ---------------------------------------------------------------------------
# Severity inference
# ---------------------------------------------------------------------------

def _infer_severity(h_type: HallucinationType, confidence: float) -> HallucinationSeverity:
    """
    Infer severity from type + confidence.

    Some types are inherently higher risk (negation flips in medical context
    are worse than a 1-year temporal error in a product description).
    This is a default heuristic — callers can override per domain.
    """
    high_risk_types = {
        HallucinationType.NEGATION_FLIP,
        HallucinationType.CONFIDENT_FABRICATION,
        HallucinationType.RELATION_ERROR,
    }
    medium_risk_types = {
        HallucinationType.ENTITY_SUBSTITUTION,
        HallucinationType.NUMERICAL_DISTORTION,
        HallucinationType.SOURCE_BLENDING,
    }

    if h_type in high_risk_types:
        base = HallucinationSeverity.HIGH if confidence >= 0.7 else HallucinationSeverity.MEDIUM
    elif h_type in medium_risk_types:
        base = HallucinationSeverity.MEDIUM if confidence >= 0.6 else HallucinationSeverity.LOW
    else:
        base = HallucinationSeverity.LOW

    return base


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate_evidence(
    all_evidence: list[tuple[HallucinationType, Evidence]],
    judge_detected_types: dict[HallucinationType, float],
) -> tuple[dict[HallucinationType, float], float, Optional[HallucinationType]]:
    """
    Combine evidence from all detectors into per-type confidence scores.

    For rule-based detectors: confidence = max evidence confidence for that type.
    For LLM judge: confidence = evidence confidence directly.

    Overall hallucination probability = 1 - product(1 - p_i) across all types.
    """
    type_confidences: dict[HallucinationType, list[float]] = {}

    for h_type, ev in all_evidence:
        if h_type not in type_confidences:
            type_confidences[h_type] = []
        type_confidences[h_type].append(ev.confidence)

    # For judge-detected types not already in evidence
    for h_type, conf in judge_detected_types.items():
        if h_type not in type_confidences:
            type_confidences[h_type] = [conf]

    # Aggregate: use max confidence per type
    detected: dict[HallucinationType, float] = {
        t: max(confs) for t, confs in type_confidences.items()
    }

    if not detected:
        return {}, 0.0, None

    # Overall probability: noisy-OR
    prob = 1.0
    for conf in detected.values():
        prob *= (1.0 - conf)
    hallucination_probability = 1.0 - prob

    dominant = max(detected.items(), key=lambda x: x[1])[0]
    return detected, hallucination_probability, dominant


# ---------------------------------------------------------------------------
# Pipeline config
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Configuration for the HallucinoType pipeline."""

    # Rule-based detectors
    use_entity_detector: bool = True
    use_temporal_detector: bool = True
    use_numerical_detector: bool = True

    # LLM judge (costs API tokens — disable for offline/cheap runs)
    use_llm_judge: bool = True
    judge_backend: str = "anthropic"      # "anthropic" | "openai"
    judge_model: Optional[str] = None

    # spaCy NER (disable if spaCy not installed)
    use_spacy: bool = True

    # Thresholds
    entity_confidence_threshold: float = 0.5
    temporal_confidence_threshold: float = 0.5
    numerical_confidence_threshold: float = 0.5
    judge_confidence_threshold: float = 0.4

    # Year tolerance for temporal detector (0 = exact match required)
    year_tolerance: int = 0


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class HallucinoTypePipeline:
    """
    End-to-end hallucination fingerprinting pipeline.

    Runs all configured detectors on a (claim, context) pair and
    returns a HallucinationFingerprint with typed labels, confidence
    scores, severity, and supporting evidence.

    Example:
        pipeline = HallucinoTypePipeline()
        fp = pipeline.run(
            claim="The study found a 78% success rate.",
            context="The trial reported a 38% improvement in the treatment group."
        )
        print(fp.summary())
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._detectors: list[BaseDetector] = []
        self._judge: Optional[LLMJudgeDetector] = None
        self._build_detectors()

    def _build_detectors(self):
        cfg = self.config

        if cfg.use_entity_detector:
            self._detectors.append(
                EntitySubstitutionDetector(
                    confidence_threshold=cfg.entity_confidence_threshold,
                    use_spacy=cfg.use_spacy,
                )
            )

        if cfg.use_temporal_detector:
            self._detectors.append(
                TemporalConfusionDetector(
                    confidence_threshold=cfg.temporal_confidence_threshold,
                    year_tolerance=cfg.year_tolerance,
                )
            )

        if cfg.use_numerical_detector:
            self._detectors.append(
                NumericalDistortionDetector(
                    confidence_threshold=cfg.numerical_confidence_threshold,
                )
            )

        if cfg.use_llm_judge:
            self._judge = LLMJudgeDetector(
                backend=cfg.judge_backend,
                model=cfg.judge_model,
                confidence_threshold=cfg.judge_confidence_threshold,
            )

    def run(
        self,
        claim: str,
        context: Optional[str] = None,
    ) -> HallucinationFingerprint:
        """
        Run the full detection pipeline on a single claim.

        Args:
            claim:   The model output to evaluate.
            context: Reference text, retrieved passages, or source document.
                     Strongly recommended. Without context, only the LLM judge
                     can detect some types, and with lower accuracy.

        Returns:
            HallucinationFingerprint with typed detections.
        """
        all_evidence: list[tuple[HallucinationType, Evidence]] = []
        judge_detected_types: dict[HallucinationType, float] = {}
        judge_response: Optional[str] = None

        # --- Rule-based detectors ---
        detector_type_map = {
            EntitySubstitutionDetector: HallucinationType.ENTITY_SUBSTITUTION,
            TemporalConfusionDetector: HallucinationType.TEMPORAL_CONFUSION,
            NumericalDistortionDetector: HallucinationType.NUMERICAL_DISTORTION,
        }

        for detector in self._detectors:
            h_type = detector_type_map.get(type(detector), detector.detects)
            try:
                evidence_list = detector.detect(claim, context)
                for ev in evidence_list:
                    all_evidence.append((h_type, ev))
            except Exception:
                # Individual detector failures shouldn't crash the pipeline
                pass

        # --- LLM judge ---
        if self._judge:
            try:
                judge_evidence = self._judge.detect(claim, context)
                judge_response = self._judge._raw_response

                # Map judge evidence back to types using their confidence
                # The judge prompt specifies type in the JSON
                raw_response = judge_response or ""
                import json, re
                cleaned = re.sub(r"```(?:json)?", "", raw_response).strip()
                try:
                    data = json.loads(cleaned)
                    for det in data.get("detected", []):
                        try:
                            h_type = HallucinationType(det.get("type", ""))
                            conf = float(det.get("confidence", 0.5))
                            if conf >= self.config.judge_confidence_threshold:
                                judge_detected_types[h_type] = max(
                                    judge_detected_types.get(h_type, 0.0), conf
                                )
                        except (ValueError, KeyError):
                            pass
                except json.JSONDecodeError:
                    pass

                for ev in judge_evidence:
                    # Assign to the most recently detected judge type
                    # (best approximation without re-parsing)
                    if judge_detected_types:
                        dominant_judge_type = max(
                            judge_detected_types.items(), key=lambda x: x[1]
                        )[0]
                        all_evidence.append((dominant_judge_type, ev))

            except Exception:
                pass

        # --- Aggregate ---
        detected, hallucination_prob, dominant = _aggregate_evidence(
            all_evidence, judge_detected_types
        )

        # --- Build severity map ---
        severity = {
            t: _infer_severity(t, c)
            for t, c in detected.items()
        }

        return HallucinationFingerprint(
            claim=claim,
            context=context,
            detected_types=detected,
            severity=severity,
            evidence=[ev for _, ev in all_evidence],
            hallucination_probability=hallucination_prob,
            dominant_type=dominant,
            judge_response=judge_response,
        )

    def run_batch(
        self,
        claims: list[str],
        contexts: Optional[list[Optional[str]]] = None,
    ) -> list[HallucinationFingerprint]:
        """
        Run the pipeline on a list of claims.

        Args:
            claims:   List of model outputs to evaluate.
            contexts: Optional list of reference texts, one per claim.
                      Pass None for a claim to run without context.
        """
        if contexts is None:
            contexts = [None] * len(claims)

        if len(contexts) != len(claims):
            raise ValueError(
                f"claims and contexts must have the same length "
                f"({len(claims)} vs {len(contexts)})"
            )

        return [
            self.run(claim, ctx)
            for claim, ctx in zip(claims, contexts)
        ]
