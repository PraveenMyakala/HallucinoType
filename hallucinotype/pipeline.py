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

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from typing import Iterable, Iterator, Optional

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
    all_evidence: list[Evidence],
) -> tuple[dict[HallucinationType, float], float, Optional[HallucinationType]]:
    """
    Combine evidence from all detectors into per-type confidence scores.

    Overall hallucination probability = 1 - product(1 - p_i) across all types.
    """
    type_confidences: dict[HallucinationType, list[float]] = {}

    for ev in all_evidence:
        type_confidences.setdefault(ev.hallucination_type, []).append(ev.confidence)

    detected: dict[HallucinationType, float] = {
        t: max(confs) for t, confs in type_confidences.items()
    }

    if not detected:
        return {}, 0.0, None

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

    use_entity_detector: bool = True
    use_temporal_detector: bool = True
    use_numerical_detector: bool = True

    use_llm_judge: bool = True
    judge_backend: str = "anthropic"
    judge_model: Optional[str] = None

    use_spacy: bool = True

    entity_confidence_threshold: float = 0.5
    temporal_confidence_threshold: float = 0.5
    numerical_confidence_threshold: float = 0.5
    judge_confidence_threshold: float = 0.4

    year_tolerance: int = 0

    # Max parallel workers for run_batch (None = ThreadPoolExecutor default)
    batch_max_workers: Optional[int] = None


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
        all_evidence: list[Evidence] = []
        judge_response: Optional[str] = None

        for detector in self._detectors:
            try:
                all_evidence.extend(detector.detect(claim, context))
            except Exception:
                pass

        if self._judge:
            try:
                judge_evidence, judge_response = self._judge.detect_and_return_raw(claim, context)
                all_evidence.extend(judge_evidence)
            except Exception:
                pass

        detected, hallucination_prob, dominant = _aggregate_evidence(all_evidence)

        severity = {
            t: _infer_severity(t, c)
            for t, c in detected.items()
        }

        return HallucinationFingerprint(
            claim=claim,
            context=context,
            detected_types=detected,
            severity=severity,
            evidence=all_evidence,
            hallucination_probability=hallucination_prob,
            dominant_type=dominant,
            judge_response=judge_response,
        )

    def run_batch(
        self,
        claims: Iterable[str],
        contexts: Optional[Iterable[Optional[str]]] = None,
    ) -> Iterator[HallucinationFingerprint]:
        """
        Yield fingerprints concurrently over a stream of claims.

        Args:
            claims:   Iterable of model outputs to evaluate.
            contexts: Optional iterable of reference texts, one per claim.

        Yields:
            HallucinationFingerprint for each (claim, context) pair, in order.
        """
        if (
            contexts is not None
            and hasattr(claims, "__len__")
            and hasattr(contexts, "__len__")
            and len(claims) != len(contexts)  # type: ignore[arg-type]
        ):
            raise ValueError("claims and contexts must have the same length")

        contexts_iter: Iterable[Optional[str]] = contexts if contexts is not None else repeat(None)

        # LLM judge calls are network I/O — CPU-based defaults (os.cpu_count()+4)
        # are too conservative. Use 32 as the floor when the judge is active.
        max_workers = self.config.batch_max_workers
        if max_workers is None and self.config.use_llm_judge:
            max_workers = 32

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            yield from executor.map(self.run, claims, contexts_iter)
