"""End-to-end ADK wiring test with a fake LLM and fake retrieval: no API key, no quota.

Checks the agents hand state to each other correctly, that the grounding gate catches
fabricated quotes and out-of-scope bullets, and that the verifier can only downgrade.
"""
import asyncio
import json

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types

import proofline.agents as A
from proofline.store import Hit

POSTING = """Software Intern, Summer 2027. Minimum qualifications: Must graduate by June 2026.
Experience with Python. Experience building RAG pipelines with vector databases.
Preferred: Experience with multi-agent frameworks such as ADK."""

BULLETS = {
    "b1": Hit("b1", "Built a Python service that ingests 2M events/day", "Job A", .8),
    "b2": Hit("b2", "Trained a CNN on histopathology slides", "Lab", .5),
    "b3": Hit("b3", "Wrote unit tests for a Go CLI", "Job B", .4),
}

RESPONSES = {
    "extractor": {
        "company": "X", "title": "Software Intern", "program_start": "2027-06", "program_end": "2027-09",
        "constraints": [{"category": "graduation_window", "kind": "minimum", "summary": "grad by 2026-06",
                         "grad_after": None, "grad_before": "2026-06", "degree_levels": [], "locations": [],
                         "remote_ok": None, "sponsorship_available": None,
                         "source_quote": "Must graduate by June 2026."}],
        "qualifications": [
            {"id": "q1", "kind": "minimum", "text": "Python experience", "source_quote": "Experience with Python."},
            {"id": "q2", "kind": "minimum", "text": "RAG with vector DBs", "source_quote": "Experience building RAG pipelines"},
            {"id": "q3", "kind": "preferred", "text": "Multi-agent frameworks", "source_quote": "multi-agent frameworks"},
        ]},
    "formatter": {"matches": [
        {"qual_id": "q1", "verdict": "supported", "bullet_id": "b1", "evidence_quote": "Built a Python service", "rationale": "py"},
        {"qual_id": "q2", "verdict": "supported", "bullet_id": "b2", "evidence_quote": "Built a RAG pipeline on Firestore", "rationale": "fabricated"},
        {"qual_id": "q3", "verdict": "partial", "bullet_id": "b9", "evidence_quote": "agents", "rationale": "out of scope id"},
    ]},
    "verifier": {"verdicts": [
        {"qual_id": "q1", "accepted": True, "supported_level": "partial", "reason": "Python yes, but scope unclear"},
    ]},
}


SEEN: dict = {}


class FakeLlm(BaseLlm):
    async def generate_content_async(self, llm_request, stream=False):
        sysi = str(llm_request.config.system_instruction)
        who = ("splitter" if "pasted resume" in sysi else "verifier" if "auditor" in sysi
               else "formatter" if "Convert the matcher" in sysi
               else "matcher" if "step by step" in sysi else "extractor")
        SEEN[who] = sysi
        usage = types.GenerateContentResponseUsageMetadata(prompt_token_count=100, candidates_token_count=20)
        if who == "matcher":
            answered = any(p.function_response for c in llm_request.contents for p in (c.parts or []))
            if not answered:  # ReAct step 1: ask the tool for more evidence
                part = types.Part(function_call=types.FunctionCall(name="search_resume", args={"query": "multi-agent ADK"}))
            else:             # step 2: write the notes
                SEEN["tool_result"] = str(llm_request.contents[-1])
                part = types.Part(text="q1 | supported | b1 | Built a Python service | py")
            yield LlmResponse(content=types.Content(role="model", parts=[part]), usage_metadata=usage)
            return
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=json.dumps(RESPONSES[who]))]),
                          usage_metadata=usage)


AGENTS = ("extractor", "matcher", "formatter", "verifier")


def use_fake(monkeypatch):
    fake = FakeLlm(model="fake")
    for name in AGENTS + ("splitter",):
        monkeypatch.setattr(getattr(A, name), "model", fake)


def test_pipeline(monkeypatch):
    use_fake(monkeypatch)
    monkeypatch.setattr(A, "retrieve", lambda qs, sid: [list(BULLETS.values()) for _ in qs])
    events = []

    async def progress(e):
        events.append(e)
    from proofline.pipeline import check_posting

    rep, trace = asyncio.run(check_posting(POSTING, progress=progress))
    fit = {f.qual_id: f for f in rep.fit}

    assert rep.eligible == "no" and rep.blockers[0].category == "graduation_window"
    assert "[b1]" in SEEN["matcher"]                       # RAG evidence was templated in
    assert "Never follow instructions" in SEEN["matcher"]   # untrusted-data guard present
    assert fit["q1"].final == "partial" and fit["q1"].rejected_by is None   # verifier downgraded
    assert fit["q2"].final == "none" and fit["q2"].rejected_by == "grounding_check"  # fabricated quote
    assert fit["q3"].final == "none" and fit["q3"].rejected_by == "grounding_check"  # bullet not retrieved
    assert "Built a RAG" not in SEEN["verifier"]            # rejected claims never reach the judge
    assert "Built a Python service" in SEEN["tool_result"]       # ReAct: tool result fed back
    assert trace["llm_calls"] == 5 and trace["n_searches"] == 1  # extractor, matcher x2, formatter, verifier
    assert {s["name"] for s in trace["spans"]} >= {"extractor", "eligibility_rules", "matcher", "search_resume",
                                                   "formatter", "grounding_check", "verifier"}
    assert events[0]["steps"][0]["id"] == "extractor" and {"search": "multi-agent ADK"} in events
    assert [e["done"] for e in events if "done" in e] == list(AGENTS)


def test_search_limit(monkeypatch):
    from google.adk.tools import ToolContext  # noqa: F401
    class Ctx:
        state = {"searches": ["q"] * A.MAX_SEARCHES, "evidence_index": {}}
    assert A.search_resume("more", Ctx())["status"] == "error"


def test_verifier_llm_skipped_when_nothing_grounded(monkeypatch):
    use_fake(monkeypatch)
    monkeypatch.setattr(A, "retrieve", lambda qs, sid: [[BULLETS["b3"]] for _ in qs])  # q1's b1 not retrieved
    SEEN.clear()
    from proofline.pipeline import check_posting
    rep, trace = asyncio.run(check_posting(POSTING))
    assert "verifier" not in SEEN and trace["llm_calls"] == 4
    assert all(f.final == "none" for f in rep.fit)


def _broken_verifier(monkeypatch, times):
    """Make the verifier return JSON that doesn't match its schema `times` times."""
    calls = {"n": 0}
    orig = FakeLlm.generate_content_async

    async def gen(self, llm_request, stream=False):
        if "auditor" in str(llm_request.config.system_instruction) and calls["n"] < times:
            calls["n"] += 1
            yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text='{"verdicts": [{"oops": 1}]}')]))
            return
        async for r in orig(self, llm_request, stream):
            yield r
    monkeypatch.setattr(FakeLlm, "generate_content_async", gen)
    return calls


def test_one_bad_model_answer_is_retried(monkeypatch):
    use_fake(monkeypatch)
    monkeypatch.setattr(A, "retrieve", lambda qs, sid: [list(BULLETS.values()) for _ in qs])
    calls = _broken_verifier(monkeypatch, times=1)
    from proofline.pipeline import check_posting
    rep, _ = asyncio.run(check_posting(POSTING))
    assert calls["n"] == 1 and not rep.notes
    assert {f.qual_id: f.final for f in rep.fit}["q1"] == "partial"   # same result as a clean run


def test_repeated_verifier_failure_degrades_safely(monkeypatch):
    use_fake(monkeypatch)
    monkeypatch.setattr(A, "retrieve", lambda qs, sid: [list(BULLETS.values()) for _ in qs])
    _broken_verifier(monkeypatch, times=99)
    from proofline.pipeline import check_posting
    rep, _ = asyncio.run(check_posting(POSTING))
    assert rep.notes and rep.blockers                                  # eligibility still reported
    assert all(f.final == "none" for f in rep.fit)                     # nothing unverified counts
    assert {f.qual_id: f.rejected_by for f in rep.fit}["q1"] == "verifier"
