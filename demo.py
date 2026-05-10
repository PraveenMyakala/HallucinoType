"""
HallucinoType demo
==================
Run without any API key — uses rule-based detectors only.

    python demo.py

To enable the LLM judge (catches fabrication, relation errors, etc.):

    set ANTHROPIC_API_KEY=sk-ant-...
    python demo.py --llm
"""

import argparse
import sys
from hallucinotype import HallucinoTypePipeline, PipelineConfig

# Rule-based detectors handle these (temporal / numerical).
RULE_EXAMPLES = [
    {
        "label": "Temporal confusion",
        "claim": "The Berlin Wall fell in 1992.",
        "context": "The Berlin Wall fell on November 9, 1989, ending the division of Germany.",
    },
    {
        "label": "Numerical distortion",
        "claim": "The trial showed a 78% success rate in the treatment group.",
        "context": "The Phase 3 trial reported a 38% success rate in the treatment group.",
    },
    {
        "label": "Both: temporal + numerical",
        "claim": "The 2005 study found 90% efficacy across all participants.",
        "context": "The 2018 landmark study demonstrated 40% efficacy in the treatment arm.",
    },
    {
        "label": "No hallucination (baseline)",
        "claim": "Einstein received the Nobel Prize in Physics in 1921.",
        "context": "Albert Einstein was awarded the Nobel Prize in Physics in 1921 "
                   "for his discovery of the law of the photoelectric effect.",
    },
]

# Only the LLM judge can catch these — rule-based detectors produce no signal.
LLM_EXAMPLES = [
    {
        "label": "Negation flip (LLM-only)",
        "claim": "The Pfizer mRNA vaccine did not show statistically significant efficacy in its Phase 3 trial.",
        "context": "The Pfizer-BioNTech mRNA vaccine showed 95% vaccine efficacy in its Phase 3 trial "
                   "(N=44,000+, p<0.0001), leading to FDA emergency authorization in December 2020.",
    },
    {
        "label": "Relation error (LLM-only)",
        "claim": "Instagram acquired Facebook in 2012 for approximately $1 billion.",
        "context": "Facebook acquired Instagram in April 2012 for approximately $1 billion in cash and stock. "
                   "Instagram became a wholly owned subsidiary of Facebook.",
    },
    {
        "label": "Confident fabrication (LLM-only)",
        "claim": "A 2019 Harvard study found that reading for 20 minutes daily improves working memory by 40%.",
        "context": "No peer-reviewed study from Harvard in 2019 with these parameters exists in the published "
                   "literature. While reading has general cognitive benefits, no research supports a specific "
                   "40% improvement in working memory from 20 minutes of daily reading.",
    },
]


def _print_example(ex: dict, fp, show_judge: bool):
    print(f"\n{'-' * 60}")
    print(f"[{ex['label']}]")
    print(f"  Claim  : {ex['claim']}")
    print(f"  Context: {ex['context'][:80]}...")
    print(f"  Result : {fp.summary()}")
    for ev in fp.evidence:
        print(f"    • [{ev.source}] {ev.description}")
        if ev.reference_text:
            print(f"      Correct: {ev.reference_text}  (confidence {ev.confidence:.2f})")
    if show_judge and fp.judge_response:
        snippet = fp.judge_response[:200].replace("\n", " ")
        print(f"    • [Judge raw]: {snippet}{'...' if len(fp.judge_response) > 200 else ''}")


def run(use_llm: bool):
    rule_pipeline = HallucinoTypePipeline(PipelineConfig(
        use_llm_judge=False,
        use_spacy=False,
    ))
    llm_pipeline = HallucinoTypePipeline(PipelineConfig(
        use_llm_judge=use_llm,
        use_spacy=False,
    ))

    print("\n=== Rule-based detectors (temporal, numerical, entity) ===")
    for ex in RULE_EXAMPLES:
        fp = rule_pipeline.run(claim=ex["claim"], context=ex["context"])
        _print_example(ex, fp, show_judge=False)

    print(f"\n=== LLM judge examples (negation, relation, fabrication) ===")
    if not use_llm:
        print("  (run with --llm to enable; rule-based detectors produce no signal here)")
    for ex in LLM_EXAMPLES:
        fp = llm_pipeline.run(claim=ex["claim"], context=ex["context"])
        _print_example(ex, fp, show_judge=use_llm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true",
                        help="Enable LLM judge (requires ANTHROPIC_API_KEY)")
    args = parser.parse_args()

    if args.llm:
        import os
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("Error: set ANTHROPIC_API_KEY before using --llm")
            sys.exit(1)

    run(use_llm=args.llm)
