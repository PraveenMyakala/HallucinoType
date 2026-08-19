"""
LangChain integration.

`HallucinoTypeCallbackHandler` attaches to any LangChain run (chain, agent,
LCEL pipeline) and fingerprints each LLM completion against retrieved
context — either passed in explicitly or captured automatically from
`on_retriever_end`. `HallucinoTypeStringEvaluator` wraps the pipeline as a
plain prediction/reference evaluator for use in LangSmith `evaluate()` runs
or ad hoc scripts. It has no LangChain base-class dependency since
`langchain.evaluation` was removed in LangChain 1.0.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from hallucinotype.pipeline import HallucinoTypePipeline, PipelineConfig
from hallucinotype.taxonomy import HallucinationFingerprint

_logger = logging.getLogger(__name__)

try:
    from langchain_core.callbacks.base import BaseCallbackHandler

    _HAS_LANGCHAIN_CALLBACKS = True
except ImportError:
    BaseCallbackHandler = object
    _HAS_LANGCHAIN_CALLBACKS = False


class HallucinoTypeCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback handler that fingerprints every LLM completion.

    Context can be supplied up front (static QA/summarization use cases) or
    left unset and captured automatically from the most recent retriever
    call in the same run (RAG chains) via `on_retriever_end`.

    Example:
        handler = HallucinoTypeCallbackHandler(threshold=0.5)
        chain.invoke({"question": "..."}, config={"callbacks": [handler]})
        for fp in handler.fingerprints:
            if fp.is_hallucinated():
                print(fp.summary())
    """

    def __init__(
        self,
        context: Optional[str] = None,
        pipeline: Optional[HallucinoTypePipeline] = None,
        config: Optional[PipelineConfig] = None,
        threshold: float = 0.5,
    ):
        if not _HAS_LANGCHAIN_CALLBACKS:
            raise ImportError(
                "langchain-core is required for HallucinoTypeCallbackHandler. "
                "Install with: pip install langchain-core"
            )
        super().__init__()
        self.static_context = context
        self.threshold = threshold
        self.pipeline = pipeline or HallucinoTypePipeline(config or PipelineConfig())
        self.fingerprints: list[HallucinationFingerprint] = []
        self._retrieved_context: Optional[str] = None

    def on_retriever_end(self, documents: Any, **kwargs: Any) -> None:
        texts = [getattr(doc, "page_content", None) for doc in documents]
        texts = [t for t in texts if t]
        if texts:
            self._retrieved_context = "\n\n".join(texts)

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        context = self.static_context or self._retrieved_context
        for generation_list in getattr(response, "generations", []):
            for generation in generation_list:
                text = getattr(generation, "text", None)
                if not text:
                    message = getattr(generation, "message", None)
                    text = getattr(message, "content", None)
                if not text:
                    continue
                fingerprint = self.pipeline.run(claim=text, context=context)
                self.fingerprints.append(fingerprint)
                if fingerprint.is_hallucinated(self.threshold):
                    _logger.warning("Possible hallucination detected: %s", fingerprint.summary())


class HallucinoTypeStringEvaluator:
    """
    Prediction/reference evaluator for LangSmith `evaluate()` runs or
    standalone scripts — no LangChain base class required.

    Example:
        evaluator = HallucinoTypeStringEvaluator()
        result = evaluator.evaluate_strings(prediction=answer, reference=context)
        # {"key": "hallucinotype", "score": 0.85, "value": "none", "comment": "..."}
    """

    def __init__(
        self,
        pipeline: Optional[HallucinoTypePipeline] = None,
        config: Optional[PipelineConfig] = None,
    ):
        self.pipeline = pipeline or HallucinoTypePipeline(config or PipelineConfig())

    def evaluate_strings(
        self,
        prediction: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        fingerprint = self.pipeline.run(claim=prediction, context=reference)
        return {
            "key": "hallucinotype",
            "score": 1.0 - fingerprint.hallucination_probability,
            "value": fingerprint.dominant_type.value if fingerprint.dominant_type else "none",
            "comment": fingerprint.summary(),
        }

    def __call__(self, run: Any, example: Any = None) -> dict:
        """LangSmith custom-evaluator entry point: `evaluate(..., evaluators=[handler])`."""
        prediction = _extract_text(getattr(run, "outputs", None))
        reference = None
        if example is not None:
            reference = _extract_text(getattr(example, "outputs", None)) or _extract_text(
                getattr(example, "inputs", None)
            )
        return self.evaluate_strings(prediction=prediction or "", reference=reference)


def _extract_text(payload: Optional[dict]) -> Optional[str]:
    if not payload:
        return None
    for key in ("output", "answer", "text", "context", "result"):
        if key in payload and isinstance(payload[key], str):
            return payload[key]
    for value in payload.values():
        if isinstance(value, str):
            return value
    return None
