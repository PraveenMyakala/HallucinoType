"""
Abstract base class for all HallucinoType detectors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from hallucinotype.taxonomy import Evidence, HallucinationType


class BaseDetector(ABC):
    """
    Base class for hallucination detectors.

    Each subclass targets one (or more) hallucination types and
    implements detect() to return a list of Evidence objects.
    """

    detects: HallucinationType

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold

    @abstractmethod
    def detect(self, claim: str, context: Optional[str] = None) -> list[Evidence]:
        """
        Analyze claim against context and return supporting evidence.

        Returns an empty list if no hallucination is detected.
        Each Evidence item explains why the detector flagged something.
        """
        ...
