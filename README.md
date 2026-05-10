# HallucinoType

**Typed hallucination detection for LLMs.**

Most hallucination detectors tell you *whether* a model hallucinated.  
HallucinoType tells you *what kind* — which changes how you fix it.

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

## Setup

```bash
git clone https://github.com/PraveenMyakala/HallucinoType.git
cd hallucinotype

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
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
python -m hallucinotype detect \
    --claim "Einstein won the Nobel Prize in 1905." \
    --context "Einstein won the Nobel Prize in Physics in 1921." \
    --no-llm --format text

# Single claim — with LLM judge (requires ANTHROPIC_API_KEY)
python -m hallucinotype detect \
    --claim "The study found 78% efficacy." \
    --context "The trial reported a 38% success rate."

# Batch from a JSONL file (one {"claim": "...", "context": "..."} per line)
python -m hallucinotype batch --input claims.jsonl --format text

# Output JSON to file
python -m hallucinotype detect --claim "..." --context "..." --output result.json
```

---

## Run Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

All 28 tests are rule-based and run in under 1 second with no API key.

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

## Roadmap

- [x] `v0.1` Core package: 8-type taxonomy, 4 detectors, typed fingerprints, 28 tests
- [x] `v0.1` CLI: `python -m hallucinotype detect/batch`
- [ ] `v0.2` Annotated benchmark dataset (typed, claim-context pairs with ground truth)
- [ ] `v0.3` Evaluation vs binary baselines (Vectara HHEM, SelfCheckGPT)
- [ ] `v0.4` Attribution: trace hallucination type to training data patterns
- [ ] `v0.5` LangChain / LlamaIndex evaluation callbacks

---

## License

Apache 2.0
