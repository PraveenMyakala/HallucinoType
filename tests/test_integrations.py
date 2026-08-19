"""
Tests for hallucinotype.integrations — LangChain / LlamaIndex eval hooks.

Skipped automatically when the optional framework isn't installed.
"""

from __future__ import annotations

import pytest

from hallucinotype.pipeline import PipelineConfig

RULE_BASED_CONFIG = PipelineConfig(use_llm_judge=False, use_spacy=False)


# ---------------------------------------------------------------------------
# LangChain
# ---------------------------------------------------------------------------

langchain_core = pytest.importorskip("langchain_core")


def _llm_result(text: str):
    from langchain_core.outputs import Generation, LLMResult

    return LLMResult(generations=[[Generation(text=text)]])


class TestHallucinoTypeCallbackHandler:
    def test_flags_hallucinated_completion(self):
        from hallucinotype.integrations.langchain import HallucinoTypeCallbackHandler

        handler = HallucinoTypeCallbackHandler(
            context="The Phase 3 trial reported a 38% success rate in the treatment group.",
            config=RULE_BASED_CONFIG,
        )
        handler.on_llm_end(_llm_result("The trial showed a 78% success rate in the treatment group."))

        assert len(handler.fingerprints) == 1
        assert handler.fingerprints[0].is_hallucinated()

    def test_captures_context_from_retriever(self):
        from hallucinotype.integrations.langchain import HallucinoTypeCallbackHandler

        class FakeDoc:
            def __init__(self, text):
                self.page_content = text

        handler = HallucinoTypeCallbackHandler(config=RULE_BASED_CONFIG)
        handler.on_retriever_end([FakeDoc("The Berlin Wall fell on November 9, 1989.")])
        handler.on_llm_end(_llm_result("The Berlin Wall fell in 1992."))

        assert len(handler.fingerprints) == 1
        assert handler.fingerprints[0].context is not None

    def test_clean_completion_not_flagged(self):
        from hallucinotype.integrations.langchain import HallucinoTypeCallbackHandler

        handler = HallucinoTypeCallbackHandler(
            context="The Berlin Wall fell on November 9, 1989.",
            config=RULE_BASED_CONFIG,
        )
        handler.on_llm_end(_llm_result("The Berlin Wall fell in 1989."))

        assert not handler.fingerprints[0].is_hallucinated()


class TestHallucinoTypeStringEvaluator:
    def test_evaluate_strings(self):
        from hallucinotype.integrations.langchain import HallucinoTypeStringEvaluator

        evaluator = HallucinoTypeStringEvaluator(config=RULE_BASED_CONFIG)
        result = evaluator.evaluate_strings(
            prediction="The trial showed a 78% success rate in the treatment group.",
            reference="The Phase 3 trial reported a 38% success rate in the treatment group.",
        )

        assert result["key"] == "hallucinotype"
        assert result["value"] == "numerical_distortion"
        assert result["score"] < 0.5


# ---------------------------------------------------------------------------
# LlamaIndex
# ---------------------------------------------------------------------------

llama_index_core = pytest.importorskip("llama_index.core")


class TestLlamaIndexEvaluator:
    def test_evaluate_flags_hallucination(self):
        from hallucinotype.integrations.llamaindex import HallucinoTypeEvaluator

        evaluator = HallucinoTypeEvaluator(config=RULE_BASED_CONFIG)
        result = evaluator.evaluate(
            query="What success rate did the trial show?",
            response="The trial showed a 78% success rate in the treatment group.",
            contexts=["The Phase 3 trial reported a 38% success rate in the treatment group."],
        )

        assert result.passing is False
        assert result.score < 0.5

    def test_evaluate_clean_response_passes(self):
        from hallucinotype.integrations.llamaindex import HallucinoTypeEvaluator

        evaluator = HallucinoTypeEvaluator(config=RULE_BASED_CONFIG)
        result = evaluator.evaluate(
            query="When did the Berlin Wall fall?",
            response="The Berlin Wall fell in 1989.",
            contexts=["The Berlin Wall fell on November 9, 1989."],
        )

        assert result.passing is True
