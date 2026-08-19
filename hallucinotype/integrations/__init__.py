"""
Thin wrappers to run HallucinoType as an eval hook inside existing
LangChain / LlamaIndex pipelines.

Not imported by `hallucinotype/__init__.py` — langchain and llama-index
are optional dependencies. Import directly:

    from hallucinotype.integrations.langchain import HallucinoTypeCallbackHandler
    from hallucinotype.integrations.llamaindex import HallucinoTypeEvaluator
"""
