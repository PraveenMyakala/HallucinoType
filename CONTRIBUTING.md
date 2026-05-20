# Contributing to HallucinoType

Thanks for your interest in contributing! Here's how to get started.

## Quick start

```bash
# 1. Fork the repo on GitHub, then clone your fork
git clone git@github.com:<your-username>/HallucinoType.git
cd HallucinoType

# 2. Create a virtual environment and install in editable mode with dev deps
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 3. Run the test suite to confirm everything works
pytest tests/ -v -m "not slow"
```

## Workflow

1. Create a branch off `main`: `git checkout -b your-feature-name`
2. Make your changes. Keep commits focused and atomic.
3. Run tests before pushing:
   ```bash
   pytest tests/ -v -m "not slow"
   ```
4. Open a pull request against `main`. Fill in the PR description with what changed and why.

## Areas to contribute

The highest-value open items (from [`CLAUDE.md`](CLAUDE.md)):

| Area | Description |
|------|-------------|
| **Benchmark dataset** | `data/benchmark_v0.jsonl` — 200–400 labeled (claim, context, type) pairs |
| **Evaluation script** | `eval.py` — compare against binary baselines (HHEM, SelfCheckGPT) |
| **New detectors** | Add a detector under `hallucinotype/detectors/` extending `BaseDetector` |
| **LangChain / LlamaIndex callbacks** | Thin wrappers so HallucinoType works as an eval hook |
| **Docs / examples** | Notebooks, usage examples, or improved README sections |

## Adding a new detector

1. Create `hallucinotype/detectors/your_detector.py` extending `BaseDetector`.
2. Register it in `pipeline.py`.
3. Add tests in `tests/test_hallucinotype.py` — use `PipelineConfig(use_llm_judge=False, use_spacy=False)` so tests run offline.
4. Update the taxonomy in `hallucinotype/taxonomy.py` if you're adding a new `HallucinationType`.

## Code style

- Python 3.10+. Type annotations everywhere.
- No `click` — CLI uses stdlib `argparse`.
- No comments explaining *what* code does — only *why* (non-obvious constraints or workarounds).
- Individual detector failures must not crash the pipeline.

## Running with the LLM judge

Set `ANTHROPIC_API_KEY` in your environment, then:

```bash
hallucinotype detect \
  --claim "Einstein won the Nobel in 1905." \
  --context "Einstein won the Nobel Prize in Physics in 1921." \
  --format text
```

## Questions?

Open a [GitHub Discussion](https://github.com/PraveenMyakala/HallucinoType/discussions) or file an issue.
