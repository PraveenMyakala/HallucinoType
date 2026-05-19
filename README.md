# HallucinoType

**Typed hallucination detection for LLMs.**

Most hallucination detectors tell you *whether* a model hallucinated.  
HallucinoType tells you *what kind* — which changes how you fix it.

[![PyPI](https://img.shields.io/pypi/v/hallucinotype)](https://pypi.org/project/hallucinotype/)
[![CI](https://github.com/PraveenMyakala/HallucinoType/actions/workflows/ci.yml/badge.svg)](https://github.com/PraveenMyakala/HallucinoType/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

---

## Hallucination Types

| Type | Description | Example |
|---|---|---|
| `entity_substitution` | Wrong entity used in place of the correct one | Attributing Einstein's Nobel Prize to Bohr |
| `temporal_confusion` | Incorrect date, year, or era | Claiming the Berlin Wall fell in 1992 |
| `source_blending` | Facts from different sources merged into one wrong claim | Two study results combined into one |
| `confident_fabrication` | Fully fabricated claim stated confidently | Citing a paper that doesn't exist |
| `numerical_distortion` | Correct context, wrong numbers | Reporting 78% efficacy when the real figure is 38% |
| `relation_error` | Correct entities, wrong relationship | "X acquired Y" when Y acquired X |
| `negation_flip` | Logical polarity inverted | "The vaccine did not show efficacy" for a trial that did |
| `overgeneralization` | Specific fact incorrectly generalized | One study's result stated as universal consensus |

---

## Install

```bash
pip install hallucinotype
```

## Setup (development)

```bash
git clone https://github.com/PraveenMyakala/HallucinoType.git
cd HallucinoType

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac/Linux

pip install -e ".[dev]"
```

---

## Run the Demo

No API key needed — rule-based detectors only:

```bash
python demo.py
```

Sample output:

```
────────────────────────────────────────────────────────────
[Temporal confusion]
  Claim  : The Berlin Wall fell in 1992.
  Context: The Berlin Wall fell on November 9, 1989...
  Result : Hallucination detected [p=0.46]: temporal_confusion (0.46)
    • Year 1992 in claim doesn't match context. Nearest year: 1989 (gap: 3 years).
      Correct: 1989  (confidence 0.46)

────────────────────────────────────────────────────────────
[Numerical distortion]
  Claim  : The trial showed a 78% success rate in the treatment group.
  Context: The Phase 3 trial reported a 38% success rate...
  Result : Hallucination detected [p=0.65]: numerical_distortion (0.65)
    • Claim uses '78' (≈78) but context has '38' (≈38). Relative error: 105.3%.
      Correct: 38%  (confidence 0.65)
```

To also run the LLM judge (detects fabrication, relation errors, negation flips):

```bash
# Windows
set ANTHROPIC_API_KEY=sk-ant-...
python demo.py --llm

# Mac/Linux
export ANTHROPIC_API_KEY=sk-ant-...
python demo.py --llm
```

---

## Use in Your Own Code

```python
from hallucinotype import HallucinoTypePipeline, PipelineConfig

# Rule-based only (no API key needed)
config = PipelineConfig(use_llm_judge=False, use_spacy=False)
pipeline = HallucinoTypePipeline(config)

fp = pipeline.run(
    claim="The study was published in 2010.",
    context="This landmark paper appeared in 2019."
)

print(fp.summary())
# Hallucination detected [p=0.58]: temporal_confusion (0.58)

print(fp.is_hallucinated())    # True
print(fp.dominant_type)        # HallucinationType.TEMPORAL_CONFUSION

for ev in fp.evidence:
    print(f"[{ev.source}] {ev.description}")
    print(f"  Correct: {ev.reference_text}  Confidence: {ev.confidence:.2f}")
```

### Configuration options

```python
# With LLM judge (Claude, default)
config = PipelineConfig(use_llm_judge=True, judge_backend="anthropic")

# With LLM judge (OpenAI)
config = PipelineConfig(use_llm_judge=True, judge_backend="openai", judge_model="gpt-4o")

# Accept up to 5-year gap before flagging temporal errors
config = PipelineConfig(year_tolerance=5)

# With spaCy NER (more accurate entity detection)
# Requires: python -m spacy download en_core_web_sm
config = PipelineConfig(use_spacy=True)
```

### Use individual detectors

```python
from hallucinotype.detectors import TemporalConfusionDetector, NumericalDistortionDetector

detector = TemporalConfusionDetector(year_tolerance=0)
evidence = detector.detect(
    claim="The paper was published in 2010.",
    context="This landmark study appeared in 2019."
)
for ev in evidence:
    print(ev.description)
```

### Batch evaluation

```python
claims = [
    "The drug showed 80% efficacy in trials.",
    "Apple acquired Microsoft in 2010.",
]
contexts = [
    "The Phase 3 trial demonstrated 40% efficacy.",
    "Apple and Microsoft have always been separate companies.",
]

results = pipeline.run_batch(claims, contexts)
for claim, fp in zip(claims, results):
    print(f"[{fp.dominant_type}] {claim}")
```

---

## Command-Line Interface

```bash
# Single claim — rule-based only (no API key needed)
hallucinotype detect \
    --claim "Einstein won the Nobel Prize in 1905." \
    --context "Einstein won the Nobel Prize in Physics in 1921." \
    --no-llm --format text

# Single claim — with LLM judge (requires ANTHROPIC_API_KEY)
hallucinotype detect \
    --claim "The study found 78% efficacy." \
    --context "The trial reported a 38% success rate."

# Batch from a JSONL file (one {"claim": "...", "context": "..."} per line)
hallucinotype batch --input claims.jsonl --format text

# Output JSON to file
hallucinotype detect --claim "..." --context "..." --output result.json
```

---

## Run Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v -m "not slow"
```

All 35 tests are rule-based and run in under 1 second with no API key.

---

## Output Schema

```
HallucinationFingerprint
├── claim                      str
├── context                    str | None
├── detected_types             dict[HallucinationType, float]   # type → confidence
├── severity                   dict[HallucinationType, Severity]
├── evidence                   list[Evidence]
├── hallucination_probability  float   # noisy-OR across all types
├── dominant_type              HallucinationType | None
└── judge_response             str | None   # raw LLM output

Evidence
├── source          str     # which detector flagged this
├── description     str     # human-readable explanation
├── span            (int, int) | None   # character offsets in claim
├── reference_text  str | None          # correct value, if known
└── confidence      float
```

---

## Architecture

```
HallucinoTypePipeline
├── EntitySubstitutionDetector   spaCy NER comparison + edit distance
├── TemporalConfusionDetector    regex year/date extraction + gap check
├── NumericalDistortionDetector  numeric extraction + window overlap + relative error
└── LLMJudgeDetector             structured prompt → Claude or GPT-4o (JSON output)
      catches: confident_fabrication, source_blending,
               relation_error, negation_flip, overgeneralization
```

Rule-based detectors run first (fast, no cost). The LLM judge handles semantically complex types that rules can't catch.

---

## Releasing

Releases are fully automated via GitHub Actions. Pushing a version tag triggers the pipeline: tests → build → publish to PyPI → GitHub Release.

### Prerequisites (one-time setup)

**1. Register a PyPI Trusted Publisher** at [pypi.org/manage/account/publishing](https://pypi.org/manage/account/publishing/):

| Field | Value |
|---|---|
| PyPI project name | `hallucinotype` |
| Owner | `PraveenMyakala` |
| Repository name | `HallucinoType` |
| Workflow filename | `release.yml` |
| Environment name | `pypi` |

**2. Create a `pypi` environment** in the GitHub repo:  
Settings → Environments → New environment → name it `pypi`.

No API tokens are stored anywhere — authentication uses OIDC.

### Cutting a release

```bash
# 1. Bump the version in both files (must match the tag exactly)
#    hallucinotype/__init__.py  →  __version__ = "0.X.0"
#    pyproject.toml             →  version = "0.X.0"

# 2. Commit, tag, push
git add hallucinotype/__init__.py pyproject.toml
git commit -m "chore: release v0.X.0"
git tag v0.X.0
git push origin main
git push origin v0.X.0
```

The pipeline then runs automatically:

```
tag push v0.X.0
  ├── test            run pytest — blocks release on failure
  ├── build           build wheel + sdist, verify tag == __version__
  ├── publish         upload to PyPI via OIDC (no token stored)
  └── github-release  attach .whl + .tar.gz to the GitHub Release
```

Monitor at: `https://github.com/PraveenMyakala/HallucinoType/actions`

### Manual trigger

If the pipeline doesn't fire (e.g. tag pushed before workflow was on `main`):

- Go to **Actions → Release to PyPI → Run workflow**

Or re-push the tag:

```bash
git push origin :refs/tags/v0.X.0   # delete remote tag
git tag -f v0.X.0                    # re-point to current commit
git push origin v0.X.0               # triggers pipeline again
```

---

## Roadmap

- [x] `v0.1` Core package: 8-type taxonomy, 4 detectors, typed fingerprints, 35 tests
- [x] `v0.2` PyPI package, automated CI/CD release pipeline
- [ ] `v0.3` Annotated benchmark dataset (typed, claim-context pairs with ground truth)
- [ ] `v0.4` Evaluation vs binary baselines (Vectara HHEM, SelfCheckGPT)
- [ ] `v0.5` LangChain / LlamaIndex evaluation callbacks

---

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
