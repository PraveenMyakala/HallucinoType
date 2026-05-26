"""
HallucinoType — Typed hallucination detection for LLMs.
"""

__version__ = "0.2.1"

from hallucinotype.pipeline import HallucinoTypePipeline, PipelineConfig
from hallucinotype.taxonomy import (
    Evidence,
    HallucinationFingerprint,
    HallucinationSeverity,
    HallucinationType,
)

__all__ = [
    "__version__",
    "HallucinationType",
    "HallucinationSeverity",
    "HallucinationFingerprint",
    "Evidence",
    "HallucinoTypePipeline",
    "PipelineConfig",
]
