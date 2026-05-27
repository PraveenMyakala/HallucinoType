"""
hallucinotype CLI
=================
Command-line interface for the HallucinoType hallucination detection pipeline.

Usage:
    hallucinotype detect --claim "..." --context "..."
    hallucinotype detect --claim "..." --no-llm
    hallucinotype batch --input claims.jsonl
    hallucinotype batch --input claims.jsonl --output results.jsonl --format text
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack
from typing import Optional

from hallucinotype import __version__
from hallucinotype.pipeline import HallucinoTypePipeline, PipelineConfig
from hallucinotype.taxonomy import HallucinationFingerprint

# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _format_json(fp: HallucinationFingerprint) -> str:
    return json.dumps(fp.to_dict(), indent=2)


def _format_text(fp: HallucinationFingerprint) -> str:
    lines = []
    lines.append(f"Claim:   {fp.claim}")
    if fp.context:
        ctx_preview = fp.context[:120] + ("..." if len(fp.context) > 120 else "")
        lines.append(f"Context: {ctx_preview}")
    lines.append("")

    if not fp.detected_types:
        lines.append("Result:  No hallucination detected.")
        return "\n".join(lines)

    lines.append(f"Result:  {fp.summary()}")
    lines.append(f"Verdict: {'HALLUCINATED' if fp.is_hallucinated() else 'CLEAN'}")
    lines.append("")

    lines.append("Detected types:")
    for h_type, conf in sorted(fp.detected_types.items(), key=lambda x: x[1], reverse=True):
        sev = fp.severity.get(h_type)
        sev_str = f"  severity={sev.value}" if sev else ""
        lines.append(f"  {h_type.value:<28} confidence={conf:.2f}{sev_str}")

    if fp.evidence:
        lines.append("")
        lines.append("Evidence:")
        for ev in fp.evidence:
            lines.append(f"  [{ev.source}] {ev.description}")
            if ev.reference_text:
                lines.append(f"    Correct value: {ev.reference_text}")
            if ev.span:
                lines.append(f"    Span: chars {ev.span[0]}-{ev.span[1]}")

    return "\n".join(lines)


def _print_result(fp: HallucinationFingerprint, fmt: str, output_file) -> None:
    if fmt == "json":
        print(_format_json(fp), file=output_file)
    else:
        print(_format_text(fp), file=output_file)


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def _build_pipeline(args) -> HallucinoTypePipeline:
    config = PipelineConfig(
        use_llm_judge=not args.no_llm,
        use_spacy=not args.no_spacy,
    )
    if hasattr(args, "backend") and args.backend:
        config.judge_backend = args.backend
    if hasattr(args, "model") and args.model:
        config.judge_model = args.model
    if hasattr(args, "year_tolerance") and args.year_tolerance is not None:
        config.year_tolerance = args.year_tolerance
    return HallucinoTypePipeline(config)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_detect(args) -> int:
    pipeline = _build_pipeline(args)
    fp = pipeline.run(claim=args.claim, context=args.context)

    with ExitStack() as stack:
        out = stack.enter_context(open(args.output, "w", encoding="utf-8")) if args.output else sys.stdout
        _print_result(fp, args.format, out)

    return 1 if fp.is_hallucinated() else 0


def cmd_batch(args) -> int:
    pipeline = _build_pipeline(args)

    claims: list[str] = []
    contexts: list[Optional[str]] = []

    try:
        with open(args.input, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"Warning: skipping line {i + 1} — {exc}", file=sys.stderr)
                    continue
                if "claim" not in r:
                    print(f"Warning: skipping record {i + 1} — missing 'claim' key", file=sys.stderr)
                    continue
                claims.append(r["claim"])
                contexts.append(r.get("context"))
    except FileNotFoundError as exc:
        print(f"Error reading input: {exc}", file=sys.stderr)
        return 2

    if not claims:
        print("Error: no valid records found in input.", file=sys.stderr)
        return 2

    n_total = 0
    n_flagged = 0

    with ExitStack() as stack:
        out = stack.enter_context(open(args.output, "w", encoding="utf-8")) if args.output else sys.stdout
        for fp in pipeline.run_batch(claims, contexts):
            n_total += 1
            if fp.is_hallucinated():
                n_flagged += 1
            if args.format == "json":
                print(json.dumps(fp.to_dict()), file=out)
            else:
                print(_format_text(fp), file=out)
                print("-" * 60, file=out)

    print(f"\n{n_flagged}/{n_total} claims flagged as hallucinated.", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hallucinotype",
        description="Typed hallucination detection for LLM outputs.",
    )
    parser.add_argument("--version", action="version", version=f"hallucinotype {__version__}")

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--no-llm", action="store_true", help="Disable LLM judge (rule-based only)")
    shared.add_argument("--no-spacy", action="store_true", help="Disable spaCy NER (regex fallback)")
    shared.add_argument("--backend", choices=["anthropic", "openai"], default="anthropic",
                        help="LLM judge backend (default: anthropic)")
    shared.add_argument("--model", default=None, help="Override judge model")
    shared.add_argument("--year-tolerance", type=int, default=None,
                        dest="year_tolerance",
                        help="Temporal detector year tolerance (default: 0)")
    shared.add_argument("--format", choices=["json", "text"], default="json",
                        help="Output format (default: json)")
    shared.add_argument("--output", "-o", default=None, help="Write output to file instead of stdout")

    sub = parser.add_subparsers(dest="command", required=True)

    # detect subcommand
    p_detect = sub.add_parser("detect", parents=[shared],
                               help="Detect hallucinations in a single claim")
    p_detect.add_argument("--claim", "-c", required=True, help="The LLM output to evaluate")
    p_detect.add_argument("--context", "-ctx", default=None,
                          help="Reference text / source document")

    # batch subcommand
    p_batch = sub.add_parser("batch", parents=[shared],
                              help="Detect hallucinations in a JSONL file of claims")
    p_batch.add_argument("--input", "-i", required=True,
                         help="Input JSONL file (one {claim, context} per line)")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "detect":
        sys.exit(cmd_detect(args))
    elif args.command == "batch":
        sys.exit(cmd_batch(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
