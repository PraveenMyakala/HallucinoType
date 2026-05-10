"""
HallucinoType Evaluation
========================
Evaluate the pipeline against benchmark_v0.jsonl and compare to baselines.

Usage:
    python eval.py                      # rule-based detectors only
    python eval.py --llm                # + LLM judge (requires ANTHROPIC_API_KEY)
    python eval.py --hhem               # + Vectara HHEM baseline (requires sentence-transformers)
    python eval.py --limit 50           # quick run on first 50 entries
    python eval.py --output results.json

Metrics reported:
    Binary  — hallucinated vs clean: precision, recall, F1, accuracy
    Typed   — per-type (one-vs-rest, relaxed): precision, recall, F1
    Confusion matrix of dominant predicted type vs ground truth
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from hallucinotype import HallucinoTypePipeline, PipelineConfig
from hallucinotype.taxonomy import HallucinationType

BENCHMARK_PATH = Path("data/benchmark_v0.jsonl")

ALL_TYPES: list[str] = [t.value for t in HallucinationType]
NONE_LABEL = "none"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_benchmark(path: Path, limit: Optional[int] = None) -> list[dict]:
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries[:limit] if limit else entries


# ---------------------------------------------------------------------------
# HallucinoType predictions
# ---------------------------------------------------------------------------

def run_hallucinotype(
    entries: list[dict],
    use_llm: bool,
    verbose: bool = True,
) -> list[dict]:
    config = PipelineConfig(use_llm_judge=use_llm, use_spacy=False)
    pipeline = HallucinoTypePipeline(config)

    results = []
    t0 = time.time()
    for i, entry in enumerate(entries):
        if verbose and (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{len(entries)}  ({elapsed:.1f}s elapsed)", end="\r", flush=True)

        fp = pipeline.run(claim=entry["claim"], context=entry["context"])

        results.append({
            "id": entry.get("id", f"row_{i}"),
            "ground_truth_type": entry["ground_truth_type"],
            "domain": entry.get("domain", ""),
            "severity": entry.get("severity"),
            "pred_dominant": fp.dominant_type.value if fp.dominant_type else NONE_LABEL,
            "pred_types": {t.value: round(c, 4) for t, c in fp.detected_types.items()},
            "hallucination_probability": round(fp.hallucination_probability, 4),
        })

    if verbose:
        print(f"  {len(entries)}/{len(entries)}  ({time.time() - t0:.1f}s elapsed)  ")
    return results


# ---------------------------------------------------------------------------
# Keyword-overlap baseline (no dependencies)
# ---------------------------------------------------------------------------

def keyword_overlap_baseline(
    entries: list[dict],
    threshold: float = 0.20,
) -> list[bool]:
    """
    Binary baseline: flag hallucination when claim/context word overlap is low.
    Stop-words are stripped; Jaccard similarity below threshold → hallucinated.
    """
    STOP = {"the", "a", "an", "is", "in", "of", "to", "and", "for", "was",
            "that", "it", "with", "by", "on", "at", "from", "his", "her",
            "its", "this", "be", "as", "or", "not", "are", "were", "has"}

    preds = []
    for e in entries:
        c_words = {w for w in e["claim"].lower().split() if w not in STOP}
        x_words = {w for w in e["context"].lower().split() if w not in STOP}
        if not c_words or not x_words:
            preds.append(False)
            continue
        jaccard = len(c_words & x_words) / len(c_words | x_words)
        preds.append(jaccard < threshold)
    return preds


# ---------------------------------------------------------------------------
# HHEM baseline (optional — needs sentence-transformers)
# ---------------------------------------------------------------------------

def try_hhem_baseline(entries: list[dict]) -> Optional[list[bool]]:
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
    except ImportError:
        print("  sentence-transformers not installed — skipping HHEM.")
        print("  Install with: pip install sentence-transformers")
        return None

    try:
        print("  Loading vectara/hallucination_evaluation_model...")
        model = CrossEncoder("vectara/hallucination_evaluation_model")
        pairs = [(e["context"], e["claim"]) for e in entries]
        scores = model.predict(pairs)
        # HHEM: score closer to 0 → hallucination (unfaithful to context)
        return [float(s) < 0.5 for s in scores]
    except Exception as exc:
        print(f"  HHEM failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


def binary_metrics(y_true: list[bool], y_pred: list[bool]) -> dict:
    tp = sum(t and p for t, p in zip(y_true, y_pred))
    fp = sum(not t and p for t, p in zip(y_true, y_pred))
    fn = sum(t and not p for t, p in zip(y_true, y_pred))
    tn = sum(not t and not p for t, p in zip(y_true, y_pred))
    p, r, f1 = _prf(tp, fp, fn)
    acc = (tp + tn) / len(y_true) if y_true else 0.0
    return {"precision": p, "recall": r, "f1": f1, "accuracy": acc,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def per_type_metrics(results: list[dict]) -> dict[str, dict]:
    """
    One-vs-rest per-type metrics using relaxed matching:
    a detection is correct if the ground-truth type appears anywhere in
    pred_types (not just as dominant_type), above the default threshold.
    """
    metrics = {}
    for htype in ALL_TYPES:
        if htype == NONE_LABEL:
            continue
        tp = fp = fn = 0
        for r in results:
            gt = r["ground_truth_type"]
            detected = htype in r["pred_types"]
            if gt == htype:
                if detected:
                    tp += 1
                else:
                    fn += 1
            else:
                if detected:
                    fp += 1

        support = sum(1 for r in results if r["ground_truth_type"] == htype)
        p, rec, f1 = _prf(tp, fp, fn)
        metrics[htype] = {
            "support": support, "precision": p, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn,
        }
    return metrics


def confusion_counts(results: list[dict]) -> dict[str, dict[str, int]]:
    """ground_truth → predicted dominant type → count."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in results:
        counts[r["ground_truth_type"]][r["pred_dominant"]] += 1
    return {k: dict(v) for k, v in counts.items()}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

SEP = "-" * 68

def _row(label: str, p: float, r: float, f1: float,
         support: int = 0, note: str = "") -> str:
    sup = f"{support:>5}" if support else "     "
    n = f"  {note}" if note else ""
    return (f"  {label:<30} {sup}  {p:>6.3f}  {r:>6.3f}  {f1:>6.3f}{n}")


def print_report(
    entries: list[dict],
    ht_results: list[dict],
    use_llm: bool,
    overlap_preds: list[bool],
    hhem_preds: Optional[list[bool]],
) -> dict:
    mode = "rule-based + LLM judge" if use_llm else "rule-based only"
    n_hal = sum(1 for e in entries if e["ground_truth_type"] != NONE_LABEL)
    n_clean = len(entries) - n_hal

    print()
    print("=" * 68)
    print("  HallucinoType Evaluation Report")
    print("=" * 68)
    print(f"  Benchmark : {BENCHMARK_PATH}  ({len(entries)} entries)")
    print(f"  Mode      : {mode}")
    print(f"  Split     : {n_hal} hallucinated  /  {n_clean} clean")
    print()

    # ---- Binary detection ----
    gt_bin = [e["ground_truth_type"] != NONE_LABEL for e in entries]
    ht_bin = [bool(r["pred_types"]) for r in ht_results]

    ht_bin_m = binary_metrics(gt_bin, ht_bin)
    ov_bin_m = binary_metrics(gt_bin, overlap_preds)

    print(f"  BINARY DETECTION  (hallucinated vs clean)")
    print(SEP)
    print(f"  {'System':<30} {'Prec':>6}  {'Recall':>6}  {'F1':>6}  {'Acc':>6}")
    print(SEP)
    print(f"  {'HallucinoType (' + mode + ')':<30} "
          f"{ht_bin_m['precision']:>6.3f}  {ht_bin_m['recall']:>6.3f}  "
          f"{ht_bin_m['f1']:>6.3f}  {ht_bin_m['accuracy']:>6.3f}")
    print(f"  {'Keyword overlap (baseline)':<30} "
          f"{ov_bin_m['precision']:>6.3f}  {ov_bin_m['recall']:>6.3f}  "
          f"{ov_bin_m['f1']:>6.3f}  {ov_bin_m['accuracy']:>6.3f}")
    if hhem_preds is not None:
        hh_m = binary_metrics(gt_bin, hhem_preds)
        print(f"  {'HHEM (Vectara, baseline)':<30} "
              f"{hh_m['precision']:>6.3f}  {hh_m['recall']:>6.3f}  "
              f"{hh_m['f1']:>6.3f}  {hh_m['accuracy']:>6.3f}")
    print(SEP)

    # ---- Per-type ----
    type_m = per_type_metrics(ht_results)

    RULE_TYPES = {"temporal_confusion", "numerical_distortion", "entity_substitution"}
    LLM_TYPES = set(ALL_TYPES) - RULE_TYPES - {NONE_LABEL}

    print()
    print(f"  PER-TYPE DETECTION  (relaxed: any detection of correct type counts)")
    print(SEP)
    print(f"  {'Type':<30} {'Supp':>5}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}")
    print(SEP)

    macro_p = macro_r = macro_f1 = 0.0
    rule_p = rule_r = rule_f1 = 0.0
    llm_p = llm_r = llm_f1 = 0.0

    for htype in ALL_TYPES:
        if htype == NONE_LABEL:
            continue
        m = type_m[htype]
        is_llm = htype in LLM_TYPES
        note = "(LLM judge)" if is_llm and not use_llm else ""
        print(_row(htype, m["precision"], m["recall"], m["f1"],
                   support=m["support"], note=note))
        macro_p += m["precision"]
        macro_r += m["recall"]
        macro_f1 += m["f1"]
        if not is_llm:
            rule_p += m["precision"]
            rule_r += m["recall"]
            rule_f1 += m["f1"]
        else:
            llm_p += m["precision"]
            llm_r += m["recall"]
            llm_f1 += m["f1"]

    n = len(ALL_TYPES) - 1  # exclude "none"
    n_rule = len(RULE_TYPES)
    n_llm = n - n_rule

    print(SEP)
    print(_row("Macro average (all types)",
               macro_p / n, macro_r / n, macro_f1 / n))
    print(_row("Rule-based types only",
               rule_p / n_rule, rule_r / n_rule, rule_f1 / n_rule))
    if use_llm:
        print(_row("LLM-judge types only",
                   llm_p / n_llm, llm_r / n_llm, llm_f1 / n_llm))
    print(SEP)

    # ---- Confusion matrix (compact: top errors only) ----
    conf = confusion_counts(ht_results)
    print()
    print(f"  DOMINANT-TYPE CONFUSION  (ground truth -> top predicted types)")
    print(SEP)
    for gt_type in ALL_TYPES:
        if gt_type not in conf:
            continue
        row = conf[gt_type]
        total = sum(row.values())
        sorted_preds = sorted(row.items(), key=lambda x: -x[1])[:3]
        parts = "  ".join(f"{p}:{c}" for p, c in sorted_preds)
        correct = row.get(gt_type, 0)
        acc = correct / total if total else 0.0
        print(f"  {gt_type:<30} (n={total:>3})  correct:{correct:>3} ({acc:.0%})  "
              f"-> {parts}")
    print(SEP)

    # ---- Clean false-positive rate ----
    fp_clean = sum(
        1 for r in ht_results
        if r["ground_truth_type"] == NONE_LABEL and r["pred_types"]
    )
    n_clean_total = sum(1 for e in entries if e["ground_truth_type"] == NONE_LABEL)
    print(f"\n  False-positive rate on clean examples: "
          f"{fp_clean}/{n_clean_total} = {fp_clean/n_clean_total:.1%}")
    print()

    return {
        "binary": ht_bin_m,
        "per_type": type_m,
        "confusion": conf,
        "fp_rate_clean": fp_clean / n_clean_total if n_clean_total else 0.0,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate HallucinoType on benchmark_v0.jsonl"
    )
    parser.add_argument(
        "--llm", action="store_true",
        help="Enable LLM judge (requires ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--hhem", action="store_true",
        help="Run Vectara HHEM baseline (requires: pip install sentence-transformers)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Evaluate only the first N entries (quick test)",
    )
    parser.add_argument(
        "--benchmark", type=Path, default=BENCHMARK_PATH,
        help=f"Path to benchmark JSONL (default: {BENCHMARK_PATH})",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Save per-entry predictions to a JSON file",
    )
    args = parser.parse_args()

    if args.llm and not __import__("os").environ.get("ANTHROPIC_API_KEY"):
        print("Error: set ANTHROPIC_API_KEY before using --llm")
        sys.exit(1)

    print(f"Loading benchmark: {args.benchmark}")
    entries = load_benchmark(args.benchmark, limit=args.limit)
    print(f"  {len(entries)} entries loaded.")

    print(f"\nRunning HallucinoType ({'rule-based + LLM judge' if args.llm else 'rule-based'})...")
    ht_results = run_hallucinotype(entries, use_llm=args.llm)

    print("Computing keyword-overlap baseline...")
    overlap_preds = keyword_overlap_baseline(entries)

    hhem_preds = None
    if args.hhem:
        print("Running HHEM baseline...")
        hhem_preds = try_hhem_baseline(entries)

    summary = print_report(entries, ht_results, args.llm, overlap_preds, hhem_preds)

    if args.output:
        payload = {
            "benchmark": str(args.benchmark),
            "mode": "llm" if args.llm else "rule-based",
            "summary": summary,
            "predictions": ht_results,
        }
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Per-entry results saved to {args.output}")


if __name__ == "__main__":
    main()
