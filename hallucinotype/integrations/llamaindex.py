"""
LlamaIndex integration.

`HallucinoTypeEvaluator` implements LlamaIndex's `BaseEvaluator` interface so
it plugs directly into `evaluate()`, `evaluate_response()`, and batch
evaluation runners (`BatchEvalRunner`) alongside built-in evaluators like
`FaithfulnessEvaluator`.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from hallucinotype.pipeline import HallucinoTypePipeline, PipelineConfig

try:
    from llama_index.core.evaluation import BaseEvaluator, EvaluationResult

    _HAS_LLAMA_INDEX = True
except ImportError:
    BaseEvaluator = object
    EvaluationResult = None
    _HAS_LLAMA_INDEX = False


class HallucinoTypeEvaluator(BaseEvaluator):
    """
    LlamaIndex `BaseEvaluator` backed by `HallucinoTypePipeline`.

    Example:
        evaluator = HallucinoTypeEvaluator(threshold=0.5)
        result = evaluator.evaluate(
            query=query,
            response=response.response,
            contexts=[node.get_content() for node in response.source_nodes],
        )
        print(result.passing, result.score, result.feedback)

    Or against a full LlamaIndex `Response` object:
        result = evaluator.evaluate_response(query=query, response=response)
    """

    def __init__(
        self,
        pipeline: Optional[HallucinoTypePipeline] = None,
        config: Optional[PipelineConfig] = None,
        threshold: float = 0.5,
    ):
        if not _HAS_LLAMA_INDEX:
            raise ImportError(
                "llama-index-core is required for HallucinoTypeEvaluator. "
                "Install with: pip install llama-index-core"
            )
        super().__init__()
        self.pipeline = pipeline or HallucinoTypePipeline(config or PipelineConfig())
        self.threshold = threshold

    def _get_prompts(self) -> dict:
        return {}

    def _update_prompts(self, prompts_dict: dict) -> None:
        pass

    async def aevaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> "EvaluationResult":
        context_text = "\n\n".join(contexts) if contexts else None
        fingerprint = self.pipeline.run(claim=response or "", context=context_text)
        return EvaluationResult(
            query=query,
            response=response,
            contexts=list(contexts) if contexts else None,
            passing=not fingerprint.is_hallucinated(self.threshold),
            score=1.0 - fingerprint.hallucination_probability,
            feedback=fingerprint.summary(),
        )
