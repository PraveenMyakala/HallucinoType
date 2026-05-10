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

EXAMPLES = [
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


def run(use_llm: bool):
    config = PipelineConfig(
        use_llm_judge=use_llm,
        use_spacy=False,  # set True if you ran: python -m spacy download en_core_web_sm
    )
    pipeline = HallucinoTypePipeline(config)

    for ex in EXAMPLES:
        fp = pipeline.run(claim=ex["claim"], context=ex["context"])
        print(f"\n{'-' * 60}")
        print(f"[{ex['label']}]")
        print(f"  Claim  : {ex['claim']}")
        print(f"  Context: {ex['context'][:80]}...")
        print(f"  Result : {fp.summary()}")
        for ev in fp.evidence:
            print(f"    • {ev.description}")
            if ev.reference_text:
                print(f"      Correct: {ev.reference_text}  (confidence {ev.confidence:.2f})")


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
