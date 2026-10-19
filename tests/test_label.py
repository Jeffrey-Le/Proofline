import json

import pytest

from eval import label
from eval.run_eval import markdown, score


def test_parse_choice():
    assert label.parse_choice("") == []
    assert label.parse_choice("1, 7 1") == ["graduation_window", "work_authorization"]
    with pytest.raises(ValueError):
        label.parse_choice("0")
    with pytest.raises(ValueError):
        label.parse_choice("x")


def test_default_verdict():
    assert label.default_verdict(["location"], []) == "no"
    assert label.default_verdict([], ["enrollment"]) == "check"
    assert label.default_verdict([], []) == "yes"


def test_key_lines_finds_eligibility():
    text = "We build things.\nMust be graduating between Dec 2026 and Aug 2027.\nFree snacks.\nVisa sponsorship is not available."
    got = label.key_lines(text)
    assert got == ["Must be graduating between Dec 2026 and Aug 2027.", "Visa sponsorship is not available."]


def test_labels_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(label, "LABELS", tmp_path / "labels.jsonl")
    label.write_labels({"a": {"id": "a", "expected_eligible": "yes"}})
    assert label.read_labels() == {"a": {"id": "a", "expected_eligible": "yes"}}


def test_markdown_table():
    def rep(eligible, blockers):
        return {"eligible": eligible, "blockers": [{"category": c} for c in blockers], "warnings": [],
                "fit": [{"final": "supported", "rejected_by": None}, {"final": "none", "rejected_by": "verifier"}]}
    tr = {"total_latency_ms": 9000, "input_tokens": 100, "output_tokens": 50, "cost_usd": 0.02}
    results = [
        ({"id": "p1", "expected_blockers": ["location"], "expected_eligible": "no"}, {"report": rep("no", ["location"]), "trace": tr}),
        ({"id": "p2", "expected_blockers": [], "expected_eligible": "yes"}, {"report": rep("no", ["degree_level"]), "trace": tr}),
    ]
    md = markdown(score(results), results)
    assert "| Blocker recall (postings with a real blocker that was caught) | 1/1 postings" in md
    assert "| False blocks (eligible postings marked not eligible) | 1/1 |" in md
    assert "false block on `p2`: degree_level" in md
    assert "$0.0200" in md


def test_markdown_lists_errors_and_single_miss():
    rep = {"eligible": "yes", "blockers": [], "warnings": [], "fit": []}
    tr = {"total_latency_ms": 1000, "input_tokens": 1, "output_tokens": 1, "cost_usd": 0.01}
    results = [({"id": "a", "expected_blockers": [], "expected_eligible": "check"}, {"report": rep, "trace": tr}),
               ({"id": "b", "expected_blockers": [], "expected_eligible": "yes"}, {"error": "TimeoutError()"})]
    md = markdown(score(results), results)
    assert "verdict on `a`: said yes, label check" in md
    assert "error on `b`: TimeoutError()" in md
    assert "- none" not in md
    assert "| False blocks (eligible postings marked not eligible) | 0/1 |" in md
