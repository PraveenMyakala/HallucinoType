"""
HallucinoType — Typed hallucination detection for LLMs.
"""

__version__ = "0.1.0"

from hallucinotype.taxonomy import (
    HallucinationType,
    HallucinationSeverity,
    HallucinationFingerprint,
    Evidence,
)
from hallucinotype.pipeline import HallucinoTypePipeline, PipelineConfig

__all__ = [
    "__version__",
    "HallucinationType",
    "HallucinationSeverity",
    "HallucinationFingerprint",
    "Evidence",
    "HallucinoTypePipeline",
    "PipelineConfig",
]
