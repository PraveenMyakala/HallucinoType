"""
HallucinoType Test Suite
========================
Tests for all detectors and the pipeline.

Run with: pytest tests/ -v
"""

import pytest
from pydantic import ValidationError

from hallucinotype import __version__
from hallucinotype.taxonomy import HallucinationType, HallucinationSeverity
from hallucinotype.detectors.temporal import TemporalConfusionDetector, extract_years
from hallucinotype.detectors.numerical import NumericalDistortionDetector, extract_numerics
from hallucinotype.pipeline import HallucinoTypePipeline, PipelineConfig


# ---------------------------------------------------------------------------
# Taxonomy tests
# ---------------------------------------------------------------------------

class TestTaxonomy:
    def test_all_types_have_values(self):
        for h_type in HallucinationType:
            assert isinstance(h_type.value, str)
            assert len(h_type.value) > 0

    def test_fingerprint_summary_no_detections(self):
        from hallucinotype.taxonomy import HallucinationFingerprint
        fp = HallucinationFingerprint(
            claim="Paris is the capital of France.",
            context="France's capital is Paris.",
        )
        assert "No hallucination" in fp.summary()

    def test_fingerprint_summary_with_detections(self):
        from hallucinotype.taxonomy import HallucinationFingerprint
        fp = HallucinationFingerprint(
            claim="Einstein won the Nobel Prize in 1905.",
            context="Einstein won the Nobel Prize in 1921.",
            detected_types={HallucinationType.TEMPORAL_CONFUSION: 0.85},
            hallucination_probability=0.85,
            dominant_hallucination_type=HallucinationType.TEMPORAL_CONFUSION,
        )
        summary = fp.summary()
        assert "temporal_confusion" in summary
        assert "0.85" in summary

    def test_fingerprint_is_hallucinated(self):
        from hallucinotype.taxonomy import HallucinationFingerprint
        fp = HallucinationFingerprint(
            claim="test",
            context=None,
            detected_types={HallucinationType.CONFIDENT_FABRICATION: 0.8},
            hallucination_probability=0.8,
        )
        assert fp.is_hallucinated(threshold=0.5) is True
        assert fp.is_hallucinated(threshold=0.9) is False

    def test_fingerprint_to_dict(self):
        from hallucinotype.taxonomy import HallucinationFingerprint
        fp = HallucinationFingerprint(
            claim="test claim",
            context="test context",
            detected_types={HallucinationType.NEGATION_FLIP: 0.7},
            hallucination_probability=0.7,
        )
        d = fp.to_dict()
        assert d["claim"] == "test claim"
        assert "negation_flip" in d["detected_types"]
        assert d["hallucination_probability"] == 0.7


# ---------------------------------------------------------------------------
# Temporal detector tests
# ---------------------------------------------------------------------------

class TestTemporalDetector:

    def setup_method(self):
        self.detector = TemporalConfusionDetector(confidence_threshold=0.3)

    def test_extract_years_basic(self):
        years = extract_years("The wall fell in 1989 and Germany reunified in 1990.")
        year_values = [y for y, _, _ in years]
        assert 1989 in year_values
        assert 1990 in year_values

    def test_extract_years_decade(self):
        years = extract_years("This happened in the 1980s during the Cold War.")
        year_values = [y for y, _, _ in years]
        assert 1980 in year_values

    def test_clear_year_mismatch(self):
        evidence = self.detector.detect(
            claim="The Berlin Wall fell in 1992.",
            context="The Berlin Wall fell on November 9, 1989."
        )
        assert len(evidence) > 0
        assert any("1992" in e.description or "1989" in e.description for e in evidence)
        assert evidence[0].reference_text == "1989"

    def test_correct_year_no_detection(self):
        evidence = self.detector.detect(
            claim="The Berlin Wall fell in 1989.",
            context="The Berlin Wall fell on November 9, 1989."
        )
        assert len(evidence) == 0

    def test_no_context_returns_empty(self):
        evidence = self.detector.detect(
            claim="Something happened in 1985.",
            context=None
        )
        assert evidence == []

    def test_year_tolerance(self):
        detector_strict = TemporalConfusionDetector(year_tolerance=0)
        detector_loose = TemporalConfusionDetector(year_tolerance=5)

        claim = "This was published in 2018."
        context = "The paper was published in 2020."

        strict_evidence = detector_strict.detect(claim, context)
        loose_evidence = detector_loose.detect(claim, context)

        assert len(strict_evidence) > 0   # 2-year gap triggers strict
        assert len(loose_evidence) == 0   # 2-year gap within 5-year tolerance

    def test_large_year_gap_high_confidence(self):
        evidence = self.detector.detect(
            claim="World War II ended in 1955.",
            context="World War II ended in 1945."
        )
        assert len(evidence) > 0
        assert evidence[0].confidence > 0.5  # 10-year gap = high confidence


# ---------------------------------------------------------------------------
# Numerical detector tests
# ---------------------------------------------------------------------------

class TestNumericalDetector:

    def setup_method(self):
        self.detector = NumericalDistortionDetector(
            confidence_threshold=0.4,
            window_overlap_threshold=0.2,
            relative_error_threshold=0.1,
        )

    def test_extract_numerics_basic(self):
        spans = extract_numerics("The study found a 78% success rate in 2,400 patients.")
        values = [s.value for s in spans]
        assert 78.0 in values
        assert 2400.0 in values

    def test_extract_numerics_millions(self):
        spans = extract_numerics("The company earned $1.5 billion in revenue.")
        assert any(s.value == 1.5e9 for s in spans)

    def test_clear_number_mismatch(self):
        evidence = self.detector.detect(
            claim="The trial showed a 78% success rate.",
            context="The trial showed a 38% success rate."
        )
        assert len(evidence) > 0
        assert any("38" in e.description or "78" in e.description for e in evidence)

    def test_correct_numbers_no_detection(self):
        evidence = self.detector.detect(
            claim="The trial showed a 38% success rate.",
            context="The trial showed a 38% success rate in 500 participants."
        )
        assert len(evidence) == 0

    def test_small_relative_error_no_detection(self):
        # 5% difference within default 15% threshold
        evidence = self.detector.detect(
            claim="Approximately 95 participants completed the study.",
            context="100 participants completed the full study protocol."
        )
        assert len(evidence) == 0

    def test_no_context_returns_empty(self):
        evidence = self.detector.detect(
            claim="There were 500 participants.",
            context=None
        )
        assert evidence == []


# ---------------------------------------------------------------------------
# Pipeline tests (no LLM calls — rule-based only)
# ---------------------------------------------------------------------------

class TestPipelineRuleBasedOnly:
    """
    Tests for the pipeline with LLM judge disabled.
    These run fast with no API calls.
    """

    def setup_method(self):
        config = PipelineConfig(
            use_llm_judge=False,
            use_spacy=False,  # Use regex fallback for CI compatibility
        )
        self.pipeline = HallucinoTypePipeline(config=config)

    def test_correct_claim_low_probability(self):
        fp = self.pipeline.run(
            claim="Einstein won the Nobel Prize in 1921.",
            context="Albert Einstein received the Nobel Prize in Physics in 1921."
        )
        assert fp.hallucination_probability < 0.5

    def test_temporal_hallucination_detected(self):
        fp = self.pipeline.run(
            claim="The study was published in 2010.",
            context="This landmark study was published in 2019."
        )
        assert fp.is_hallucinated()
        assert HallucinationType.TEMPORAL_CONFUSION in fp.detected_types

    def test_numerical_hallucination_detected(self):
        fp = self.pipeline.run(
            claim="The model achieved 95% accuracy on the benchmark.",
            context="The model achieved 45% accuracy on this benchmark."
        )
        assert fp.is_hallucinated()
        assert HallucinationType.NUMERICAL_DISTORTION in fp.detected_types

    def test_no_context_returns_fingerprint(self):
        fp = self.pipeline.run(
            claim="Some claim without any context provided.",
            context=None
        )
        assert fp is not None
        assert fp.claim == "Some claim without any context provided."

    def test_batch_run(self):
        claims = [
            "The study found 78% efficacy.",
            "The treatment was approved in 1999.",
        ]
        contexts = [
            "The study found 38% efficacy in the treatment group.",
            "The FDA approved the treatment in 2010.",
        ]
        results = self.pipeline.run_batch(claims, contexts)
        assert len(results) == 2
        assert all(isinstance(r, type(results[0])) for r in results)

    def test_batch_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            self.pipeline.run_batch(["claim1", "claim2"], ["context1"])

    def test_dominant_type_is_highest_confidence(self):
        fp = self.pipeline.run(
            claim="The experiment in 2005 showed 90% success.",
            context="The 2018 experiment showed 40% success."
        )
        if fp.detected_types:
            dominant = fp.dominant_type
            max_conf = max(fp.detected_types.values())
            assert fp.detected_types.get(dominant, 0) == max_conf

    def test_severity_assigned_for_detected_types(self):
        fp = self.pipeline.run(
            claim="The study was conducted in 1980.",
            context="The landmark study was conducted in 2020."
        )
        for h_type in fp.detected_types:
            assert h_type in fp.severity
            assert isinstance(fp.severity[h_type], HallucinationSeverity)


# ---------------------------------------------------------------------------
# Evidence tests
# ---------------------------------------------------------------------------

class TestEvidence:
    def test_confidence_validation(self):
        from hallucinotype.taxonomy import Evidence
        with pytest.raises(ValidationError):
            Evidence(hallucination_type=HallucinationType.TEMPORAL_CONFUSION, source="test", description="test", confidence=1.5)
        with pytest.raises(ValidationError):
            Evidence(hallucination_type=HallucinationType.TEMPORAL_CONFUSION, source="test", description="test", confidence=-0.1)

    def test_valid_evidence(self):
        from hallucinotype.taxonomy import Evidence
        ev = Evidence(
            hallucination_type=HallucinationType.TEMPORAL_CONFUSION,
            source="TestDetector",
            description="Year mismatch detected.",
            span=(10, 14),
            reference_text="1989",
            confidence=0.85,
        )
        assert ev.confidence == 0.85
        assert ev.reference_text == "1989"


# ---------------------------------------------------------------------------
# CLI tests  (python -m hallucinotype)
# ---------------------------------------------------------------------------

class TestCLI:

    def _run(self, *args):
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-m", "hallucinotype", *args],
            capture_output=True,
            text=True,
        )
        return result.returncode, result.stdout, result.stderr

    def test_detect_json_output(self):
        import json
        rc, out, _ = self._run(
            "detect",
            "--claim", "The study was published in 2005.",
            "--context", "This landmark paper appeared in 2019.",
            "--no-llm", "--no-spacy", "--format", "json",
        )
        data = json.loads(out)
        assert "hallucination_probability" in data
        assert "detected_types" in data

    def test_detect_text_output(self):
        _, out, _ = self._run(
            "detect",
            "--claim", "The trial showed 78% efficacy.",
            "--context", "The trial showed 38% efficacy.",
            "--no-llm", "--no-spacy", "--format", "text",
        )
        assert "Claim:" in out

    def test_exit_code_hallucinated(self):
        rc, _, _ = self._run(
            "detect",
            "--claim", "The Berlin Wall fell in 1999.",
            "--context", "The Berlin Wall fell in 1989.",
            "--no-llm", "--no-spacy",
        )
        assert rc == 1

    def test_exit_code_clean(self):
        rc, _, _ = self._run(
            "detect",
            "--claim", "Paris is the capital of France.",
            "--context", "The capital of France is Paris.",
            "--no-llm", "--no-spacy",
        )
        assert rc == 0

    def test_batch_from_jsonl(self, tmp_path):
        import json
        input_file = tmp_path / "claims.jsonl"
        records = [
            {"claim": "The study was published in 2005.", "context": "Published in 2019."},
            {"claim": "Paris is the capital of France.", "context": "France's capital is Paris."},
        ]
        input_file.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        rc, out, err = self._run(
            "batch", "--input", str(input_file),
            "--no-llm", "--no-spacy", "--format", "json",
        )
        lines = [l for l in out.strip().split("\n") if l.strip()]
        assert len(lines) == 2
        for line in lines:
            data = json.loads(line)
            assert "hallucination_probability" in data
        assert "1/2 claims" in err

    def test_version_flag(self):
        rc, out, err = self._run("--version")
        assert rc == 0
        assert __version__ in (out + err)

    def test_help_shows_subcommands(self):
        rc, out, _ = self._run("--help")
        assert rc == 0
        assert "detect" in out
        assert "batch" in out
