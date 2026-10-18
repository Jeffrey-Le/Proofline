"""Visitor sessions end to end with a fake LLM, fake embeddings and a real local LanceDB."""
import asyncio
import hashlib
import math

import pytest

import proofline.agents as A
import proofline.pipeline as P
import proofline.store as S
from proofline import config
from tests.test_pipeline_fake import POSTING, RESPONSES, SEEN, use_fake

RESUME = """Jane Doe | jane@example.com | 555-0100
Experience
Software Intern, Acme Corp
 • Built a retrieval service in Python that serves 3M queries a day
 • Ran on-call for the team's Kubernetes cluster for two quarters
Skills: Python, Go
"""

SPLIT = {"bullets": [
    {"text": "Built a retrieval service in Python that serves 3M queries a day", "source": "Software Intern, Acme Corp"},
    {"text": "Ran on-call for the team's Kubernetes cluster for two quarters", "source": "Software Intern, Acme Corp"},
    {"text": "Led a team of 40 engineers to ship a RAG platform", "source": "Acme"},            # fabricated
    {"text": "Built a retrieval service in Python that serves 3M queries a day", "source": "dup"},  # duplicate
    {"text": "Python, Go", "source": "Skills"},                                                   # too short
]}

PROFILE = {"graduation": "2026-05", "degree_level": "BS", "fields": ["Computer Science"],
           "willing_locations": ["Remote"], "open_to_relocation": True, "needs_sponsorship": False,
           "next_program": None}


def fake_embed(texts, task):
    out = []
    for t in texts:
        h = hashlib.sha256(t.encode()).digest()
        v = [b / 255 for b in h[:8]]
        n = math.sqrt(sum(x * x for x in v))
        out.append([x / n for x in v])
    return out


@pytest.fixture
def env(monkeypatch, tmp_path):
    store = S.LanceStore(str(tmp_path / "db"))
    for mod in (S, P):
        monkeypatch.setattr(mod, "get_store", lambda: store)
        monkeypatch.setattr(mod, "embed", fake_embed)
    use_fake(monkeypatch)
    monkeypatch.setitem(RESPONSES, "splitter", SPLIT)
    store.put_session("owner", [{"id": "o1", "text": "Owner bullet about Rust services", "source": "AWS",
                                 "vector": fake_embed(["x"], "")[0]}], None, None)
    return store


def test_session_keeps_only_verbatim_unique_bullets(env):
    out, log = asyncio.run(P.create_session(RESUME, PROFILE))
    assert [b["text"] for b in out["bullets"]] == [SPLIT["bullets"][0]["text"], SPLIT["bullets"][1]["text"]]
    assert out["dropped"] == 3 and len(out["session_id"]) == 32
    assert "jane@example.com" not in str(env.search(fake_embed(["q"], "")[0], 10, out["session_id"]))


def test_sessions_are_isolated(env):
    out, _ = asyncio.run(P.create_session(RESUME, PROFILE))
    q = fake_embed(["q"], "")[0]
    assert {h.id for h in env.search(q, 10, "owner")} == {"o1"}
    assert {h.id for h in env.search(q, 10, out["session_id"])} == {"b01", "b02"}


def test_check_uses_visitor_profile_and_evidence(env):
    out, _ = asyncio.run(P.create_session(RESUME, PROFILE))
    SEEN.clear()
    rep, _ = asyncio.run(P.check_posting(POSTING, session_id=out["session_id"]))
    # Visitor graduates 2026-05, so the "graduate by June 2026" rule passes for them.
    assert not rep.blockers and rep.eligibility[0].status == "pass"
    assert "Built a retrieval service" in SEEN["matcher"] and "Owner bullet" not in SEEN["matcher"]


def test_expired_and_deleted_sessions(env, monkeypatch):
    out, _ = asyncio.run(P.create_session(RESUME, PROFILE))
    sid = out["session_id"]
    monkeypatch.setattr(config, "SESSION_TTL_HOURS", -1)
    expired, _ = asyncio.run(P.create_session(RESUME, PROFILE))
    with pytest.raises(P.SessionExpired):
        asyncio.run(P.check_posting(POSTING, session_id=expired["session_id"]))
    P.delete_session(sid)
    with pytest.raises(P.SessionExpired):
        asyncio.run(P.check_posting(POSTING, session_id=sid))
    with pytest.raises(ValueError):
        P.delete_session("owner")


def test_bad_session_ids_rejected():
    for bad in ["../etc", "x' OR 1=1 --", "", "ABC"]:
        with pytest.raises(ValueError):
            S.check_sid(bad)


def test_owner_profile_comes_from_the_database(env):
    """Deployed builds contain no profile.yaml; the owner's profile is read from the store."""
    rows = [{"id": "o1", "text": "Owner bullet about Rust services", "source": "AWS",
             "vector": fake_embed(["x"], "")[0]}]
    env.put_session("owner", rows, {**PROFILE, "graduation": "2026-05"}, None)
    rep, _ = asyncio.run(P.check_posting(POSTING, session_id="owner"))
    assert rep.eligibility[0].status == "pass"  # profile.yaml says Dec 2026, which would block
