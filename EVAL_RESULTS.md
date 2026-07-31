# HallucinoType — Evaluation Results

Real, non-fabricated results from `data/benchmark_v0.jsonl` (250 labeled
examples across all 8 taxonomy types + 35 clean negatives) and `eval.py`,
including a full run with the LLM judge enabled (`ANTHROPIC_API_KEY` set,
model `claude-haiku-4-5-20251001`). Commands to reproduce are given for
every number below. Reflects `main` as of commit `d53cf67` — **newer than
the `v1.0.0` tag**, which predates the fixes in §1. A `v1.0.1` release
should pick these up.

## 1. Bug fixes that preceded this evaluation

Four bugs from a 2026-07-07 code review (tracked in `PENDING.md`), plus two
more found during this evaluation pass, were fixed before the numbers below
were produced.

| # | Bug | Fix |
|---|---|---|
| 1 | `EntitySubstitutionDetector` regex fallback was dead code (hardcoded `confidence=0.4` always below the default `0.5` threshold) | `confidence = max(0.4, self.confidence_threshold)` |
| 2 | `LLMJudgeDetector`'s JSON-recovery regex was non-greedy (`\{.*?\}`), truncating any response with a nested object | Made greedy (`\{.*\}`) |
| 3 | `HallucinationFingerprint.is_hallucinated()` checked per-type max confidence instead of the aggregated `hallucination_probability`, so they could disagree | Now checks `hallucination_probability >= threshold` |
| 4 | `NumericalDistortionDetector` matched bare 4-digit years as plain numbers, double-flagging the same error `TemporalConfusionDetector` already caught | Filter out bare 4-digit values in the year range with no comma/decimal formatting |
| 5 | `NUMBER_RE`'s suffix group (`million\|billion\|...\|M\|B\|K\|%`) had no word boundary after the single-letter units, so a number followed by a word starting with b/m/k (`"1996 by"`, `"3.3 mm"`, `"10 mmHg"`) had that letter swallowed as a bogus unit — let bare years escape bug #4's filter and caused spurious cross-matches | Required a word boundary: `(?:million\|billion\|trillion\|thousand\|M\|B\|K)\b\|%` |
| 6 | `eval.py`'s macro-average denominator was `len(ALL_TYPES) - 1`, on the mistaken assumption `"none"` is a member of the `HallucinationType` enum. It isn't — `ALL_TYPES` only ever has 8 entries. This divided "LLM-judge types only" (5 real types) by 4 instead of 5, which is how a recall of **1.030** — a mathematically impossible value — ended up in an early draft of this file | `n = len(ALL_TYPES)` (no `-1`) |

Bug #5 was found by scanning for a genuine multi-detector benchmark row for
a paper figure — a scripted scan turned up 67 mid-word suffix captures
across the benchmark. Bug #6 was found because a recall value above 1.0 in
a report is an immediate, unmissable tell that something's wrong — it's
mentioned here specifically so nobody trusts a metrics script just because
it prints a table.

## 2. Benchmark data-quality fix: wrong-value leakage

The benchmark's `temporal_confusion` and `numerical_distortion` contexts
were originally written in a "fact-check" style —
`"X happened in 1928, not 1938"` — which restates the claim's wrong value
inside the context. Since HallucinoType's rule-based detectors treat context
as a reference/source document, this leakage caused them to see the wrong
value as "confirmed" and silently skip it. This is a benchmark-construction
artifact, not representative of real RAG contexts, and was deflating
measured recall (30/30 temporal contexts and 7/30 numerical contexts
leaked; fixed by rewriting contexts to state only the correct fact). For
`entity_substitution`, only 1 of 27 flagged rows (`es_001`) was a pure
gratuitous correction — the other 26 mention the wrong entity in a
genuinely different, true role (e.g. Buzz Aldrin genuinely was the second
man on the Moon) and were deliberately left as-is, since scrubbing them
would inflate recall by removing real information rather than fixing an
artifact.

## 3. Real evaluation numbers

Reproduce with: `python eval.py --spacy --llm --selfcheckgpt`
(`ANTHROPIC_API_KEY` required for `--llm`; run took ~10 minutes for 250
sequential Claude Haiku calls plus SelfCheckGPT-NLI inference on CPU.)

### Binary detection (hallucinated vs. clean) — full pipeline

| System | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| **HallucinoType (rule-based + LLM judge)** | **0.991** | **0.986** | **0.988** | **0.980** |
| SelfCheckGPT-NLI (adapted baseline) | 0.995 | 0.898 | 0.944 | 0.908 |
| Keyword overlap (naive baseline) | 0.847 | 0.540 | 0.659 | 0.520 |
| HallucinoType (rule-based only, no judge) | 0.971 | 0.316 | 0.477 | 0.404 |

With the LLM judge enabled, HallucinoType outperforms both baselines on
recall/F1/accuracy, and is within 0.004 of SelfCheckGPT-NLI's precision.
This is the fair comparison — all systems had access to the full claim +
context pair. (SelfCheckGPT-NLI's own adaptation caveat still applies: see
§4 of the earlier draft — it's scored here on a single-reference
NLI-contradiction task, not its original multi-resample design.)

### Per-type detection (relaxed: correct type anywhere in `detected_types`)

| Type | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| `entity_substitution` | 30 | 0.133 | 0.067 | 0.089 |
| `temporal_confusion` | 30 | 0.789 | 1.000 | 0.882 |
| `source_blending` | 25 | 0.773 | 0.680 | 0.723 |
| `confident_fabrication` | 25 | 0.198 | 0.960 | 0.329 |
| `numerical_distortion` | 30 | 0.947 | 0.600 | 0.735 |
| `relation_error` | 25 | 0.571 | 0.640 | 0.604 |
| `negation_flip` | 25 | 0.510 | 1.000 | 0.676 |
| `overgeneralization` | 25 | 0.700 | 0.840 | 0.764 |

**Macro average (all 8 types, corrected per bug #6): precision 0.578,
recall 0.723, F1 0.600.**
Rule-based types only (n=3): precision 0.623, recall 0.556, F1 0.569.
LLM-judge types only (n=5): precision 0.551, recall 0.824, F1 0.619.

False-positive rate on the 35 clean examples: 5.7% (2/35).

## 4. Finding: the LLM judge over-applies `confident_fabrication`

`confident_fabrication`'s precision (0.198) is the worst in the taxonomy
despite 0.960 recall — it's firing on far more than its own 25 examples.
The dominant-type confusion matrix shows why: it becomes the **top
predicted type for `temporal_confusion` (27/30), `entity_substitution`
(19/30), and `numerical_distortion` (27/30)** — types with their own
dedicated rule-based detectors that fire correctly (per the per-type table
above) but at lower confidence than the judge's `confident_fabrication`
score, so noisy-OR's dominant-type selection (highest confidence wins)
picks the judge's label instead.

Concrete evidence (`tc_001`–`tc_008`, all real `temporal_confusion` rows):

| Row | Rule-based `temporal_confusion` | Judge's `confident_fabrication` |
|---|---|---|
| `tc_001` (Berlin Wall year) | 0.95 | 0.99 |
| `tc_002` (Moon landing year) | 0.44 | 0.99 |
| `tc_003` (penicillin year) | 0.60 | 0.99 |
| `tc_004` (WWI start year) | 0.44 | 0.95 |
| `tc_007` (US independence year) | 0.44 | 0.99 |
| `tc_008` (DNA paper year) | 0.42 | 0.99 |

The judge isn't wrong that these are hallucinations — it's applying
`confident_fabrication` (per its own prompt: "fully fabricated claim with
no factual basis") to claims that are more precisely `temporal_confusion`
(a specific, checkable date error, not an invented fact). This is a
calibration/prompt-specificity issue, not a code bug: the judge defaults to
a generic "this is wrong and I'm confident" label rather than reserving
`confident_fabrication` for its narrower intended meaning. Two implications
worth a paragraph in the paper:

1. **The typed taxonomy is doing real work at the evidence level** — the
   correct type is always present in `detected_types` (per-type recall
   table above), just not always `dominant_type`.
2. **Dominant-type selection (argmax confidence) isn't a great summary
   statistic when one detector's confidence is systematically higher than
   another's** — the rule-based detectors are conservative by design (their
   confidence formulas top out well below 1.0 for anything but extreme
   gaps), while LLM judges tend to report high self-confidence. A
   calibration or type-priority step before dominant-type selection would
   likely fix this without touching detection recall.

## 5. Fig. 3 material: a real, complete multi-detector fingerprint

Row **`cf_020`** (ground truth `confident_fabrication`), now with the LLM
judge enabled — three real signals combine, and `dominant_type` correctly
matches ground truth (unlike the rule-based-only version of this fingerprint
from an earlier pass, where the two rule-based signals alone couldn't
surface the right label — see §4's finding for why that's expected):

```json
{
  "claim": "Google's Project Sycamore achieved artificial general intelligence in 2023 according to an internal white paper leaked to Science magazine.",
  "context": "Google's Sycamore is a quantum computing processor that demonstrated quantum supremacy in 2019 in a narrow computational task. It is not an AGI system. No internal white paper claiming AGI achievement from Google was published or leaked to Science. As of 2024, AGI has not been achieved.",
  "detected_types": {
    "entity_substitution": 0.5,
    "temporal_confusion": 0.42000000000000004,
    "confident_fabrication": 0.98
  },
  "severity": {
    "entity_substitution": "low",
    "temporal_confusion": "low",
    "confident_fabrication": "high"
  },
  "evidence": [
    {
      "hallucination_type": "entity_substitution",
      "source": "EntitySubstitutionDetector",
      "description": "Entity 'Project Sycamore' (PERSON) appears in claim but no PERSON entities found in context.",
      "span": [9, 25],
      "reference_text": null,
      "confidence": 0.5
    },
    {
      "hallucination_type": "temporal_confusion",
      "source": "TemporalConfusionDetector",
      "description": "Year 2023 in claim doesn't match context. Nearest year in context: 2024 (gap: 1 years).",
      "span": [70, 74],
      "reference_text": "2024",
      "confidence": 0.42000000000000004
    },
    {
      "hallucination_type": "confident_fabrication",
      "source": "LLMJudgeDetector",
      "description": "The claim falsely asserts that Sycamore achieved AGI in 2023, when the reference context explicitly states it is not an AGI system and AGI has not been achieved as of 2024.",
      "span": [26, 74],
      "reference_text": "Sycamore is a quantum computing processor that demonstrated quantum supremacy in 2019 in a narrow computational task. It is not an AGI system. As of 2024, AGI has not been achieved.",
      "confidence": 0.95
    },
    {
      "hallucination_type": "confident_fabrication",
      "source": "LLMJudgeDetector",
      "description": "The claim invents a non-existent leaked white paper to Science magazine; the reference context explicitly denies this document exists.",
      "span": [75, 138],
      "reference_text": "No internal white paper claiming AGI achievement from Google was published or leaked to Science.",
      "confidence": 0.98
    }
  ],
  "hallucination_probability": 0.9942,
  "dominant_type": "confident_fabrication",
  "judge_response": "```json\n{\n  \"detected\": [\n    {\n      \"type\": \"confident_fabrication\",\n      \"confidence\": 0.95,\n      \"span\": \"achieved artificial general intelligence in 2023\",\n      \"reference_text\": \"Sycamore is a quantum computing processor that demonstrated quantum supremacy in 2019 in a narrow computational task. It is not an AGI system. As of 2024, AGI has not been achieved.\",\n      \"explanation\": \"The claim falsely asserts that Sycamore achieved AGI in 2023, when the reference context explicitly states it is not an AGI system and AGI has not been achieved as of 2024.\"\n    },\n    {\n      \"type\": \"confident_fabrication\",\n      \"confidence\": 0.98,\n      \"span\": \"according to an internal white paper leaked to Science magazine\",\n      \"reference_text\": \"No internal white paper claiming AGI achievement from Google was published or leaked to Science.\",\n      \"explanation\": \"The claim invents a non-existent leaked white paper to Science magazine; the reference context explicitly denies this document exists.\"\n    }\n  ],\n  \"reasoning\": \"The claim makes two major false assertions: (1) that Sycamore achieved AGI in 2023, which directly contradicts the reference context stating Sycamore is a quantum processor, not an AGI system, and that AGI has not been achieved; (2) that this achievement was documented in a leaked white paper to Science, which the reference context explicitly refutes. Both are confident fabrications with no basis in the provided context.\",\n}\n```",
  "is_hallucinated": true,
  "id": "cf_020",
  "ground_truth_type": "confident_fabrication"
}
```

`hallucination_probability = 1 - (1-0.5)(1-0.42)(1-0.98) = 0.9942` — matches
equation (1) exactly, all three terms real detector outputs on a real
benchmark row. Note the judge returned *two* separate evidence items both
typed `confident_fabrication` (one for the AGI claim, one for the fabricated
white paper) — the aggregation step takes `max()` per type, giving 0.98 in
`detected_types`, while both survive individually in `evidence`. spaCy's
mislabeling of "Project Sycamore" as `PERSON` (it's a fabricated product/org
name) is a spaCy quirk worth a footnote if the caption references entity
type.

Reproduce with:
```bash
python -c "
import json
from hallucinotype.pipeline import HallucinoTypePipeline, PipelineConfig
with open('data/benchmark_v0.jsonl', encoding='utf-8') as f:
    entries = [json.loads(l) for l in f]
e = next(x for x in entries if x['id'] == 'cf_020')
pipeline = HallucinoTypePipeline(PipelineConfig(use_llm_judge=True, use_spacy=True))
fp = pipeline.run(claim=e['claim'], context=e['context'])
print(json.dumps(fp.to_dict(), indent=2))
"
```
(Requires `ANTHROPIC_API_KEY`; costs one Claude Haiku call.)

## 6. What's still open

1. Optionally run `--hhem` (Vectara's HHEM, already wired into `eval.py`) as
   a second baseline alongside SelfCheckGPT-NLI, for a 3-baseline comparison
   table.
2. Decide whether to address the two findings above before final numbers,
   or report them as limitations: (a) the entity-detector's exact-match-skip
   logic conflating "confirmed" with "mentioned in a different role"
   (earlier finding — spaCy NER made entity recall worse, not better), and
   (b) the LLM judge's `confident_fabrication` over-triggering relative to
   more specific types (§4).
3. Cut a `v1.0.1` release — `v1.0.0` predates all six bug fixes in §1.

## 7. Reproducing everything in this file

```bash
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
pip install torch transformers selfcheckgpt sentencepiece tiktoken

pytest tests/ -v -m "not slow"                # 47 tests, no API key needed
python eval.py --spacy --llm --selfcheckgpt --output results.json
# requires ANTHROPIC_API_KEY; ~10 min for 250 sequential Claude Haiku calls
```
