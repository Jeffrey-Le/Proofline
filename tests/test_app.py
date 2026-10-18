import app as A
from fastapi.testclient import TestClient

from proofline import config


def reset():
    A._ip_hits.clear()
    A._spend.update(date=None, usd=0.0, requests=0)


def test_limits_and_errors(monkeypatch):
    async def fake(text, session_id="owner", cloud_trace=None, progress=None):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")
    monkeypatch.setattr(A, "check_posting", fake)
    monkeypatch.setattr(A, "PER_IP_PER_HOUR", 2)
    reset()
    c = TestClient(A.app)
    assert c.get("/").status_code == 200
    assert c.post("/check", json={"posting": "short"}).status_code == 400
    assert c.post("/check", json={"posting": "x" * 100, "session_id": "../x"}).status_code == 400
    body, h = {"posting": "x" * 100}, {"x-forwarded-for": "1.2.3.4"}
    assert c.post("/check", json=body, headers=h).status_code == 503
    assert c.post("/check", json=body, headers=h).status_code == 503
    assert c.post("/check", json=body, headers=h).status_code == 429  # third in the hour


def test_daily_budget_pauses_public_but_not_owner(monkeypatch):
    class R:
        def model_dump(self): return {}
    async def fake(text, session_id="owner", cloud_trace=None, progress=None):
        return R(), {"cost_usd": 0.30}
    monkeypatch.setattr(A, "check_posting", fake)
    monkeypatch.setattr(A, "API_TOKEN", "secret")
    monkeypatch.setattr(config, "DAILY_BUDGET_USD", 0.50)
    reset()
    c, body = TestClient(A.app), {"posting": "x" * 100}
    assert c.post("/check", json=body, headers={"x-forwarded-for": "1.1.1.1"}).status_code == 200  # $0.30
    assert c.post("/check", json=body, headers={"x-forwarded-for": "2.2.2.2"}).status_code == 200  # $0.60
    r = c.post("/check", json=body, headers={"x-forwarded-for": "3.3.3.3"})
    assert r.status_code == 429 and "daily limit" in r.json()["detail"]
    owner = c.post("/check", json=body, headers={"x-api-token": "secret"})
    assert owner.status_code == 200 and owner.json()["trace"] is not None
    assert c.post("/check", json=body, headers={"x-api-token": "wrong", "x-forwarded-for": "9.9.9.9"}).status_code == 429
    assert c.get("/healthz").json()["spent_today_usd"] == 0.9
    assert c.get("/privacy").status_code == 200


def test_session_validation():
    reset()
    c = TestClient(A.app)
    bad = {"resume": "x" * 300, "profile": {"graduation": "June 2026", "degree_level": "BS", "fields": [],
           "willing_locations": [], "open_to_relocation": True, "needs_sponsorship": False}}
    assert c.post("/session", json=bad).status_code == 422
    assert c.delete("/session/owner").status_code == 400


def test_stream_emits_progress_then_result(monkeypatch):
    import json as _json

    class R:
        def model_dump(self): return {"eligible": "yes"}

    async def fake(text, session_id="owner", cloud_trace=None, progress=None):
        await progress({"steps": [{"id": "extractor", "label": "Reading the posting"}]})
        await progress({"search": "kubernetes"})
        await progress({"done": "extractor"})
        return R(), {"cost_usd": 0.01}
    monkeypatch.setattr(A, "check_posting", fake)
    reset()
    with TestClient(A.app).stream("POST", "/check/stream", json={"posting": "x" * 100}) as r:
        lines = [_json.loads(l) for l in r.iter_lines() if l]
    assert lines[1] == {"search": "kubernetes"} and lines[-1]["result"]["report"] == {"eligible": "yes"}
    assert lines[-1]["result"]["trace"] is None  # timings and cost are for the owner only


def test_stream_reports_errors(monkeypatch):
    import json as _json

    async def fake(text, session_id="owner", cloud_trace=None, progress=None):
        raise RuntimeError("RESOURCE_EXHAUSTED")
    monkeypatch.setattr(A, "check_posting", fake)
    reset()
    with TestClient(A.app).stream("POST", "/check/stream", json={"posting": "x" * 100}) as r:
        last = [_json.loads(l) for l in r.iter_lines() if l][-1]
    assert last["status"] == 503


def test_real_resume_needs_a_private_link(monkeypatch):
    class R:
        def model_dump(self): return {}
    seen = []

    async def fake(text, session_id="owner", cloud_trace=None, progress=None):
        seen.append(session_id)
        return R(), {"cost_usd": 0.0}
    monkeypatch.setattr(A, "check_posting", fake)
    monkeypatch.setattr(A, "API_TOKEN", "owner-secret")
    monkeypatch.setattr(A, "SHARE_TOKEN", "share-secret")
    reset()
    c, body = TestClient(A.app), {"posting": "x" * 100, "session_id": "owner"}
    assert c.post("/check", json=body).status_code == 403
    assert c.post("/check", json=body, headers={"x-share-token": "guess"}).status_code == 403
    r = c.post("/check", json=body, headers={"x-share-token": "share-secret"})
    assert r.status_code == 200 and r.json()["trace"] is None          # recruiters see no diagnostics
    assert c.post("/check", json=body, headers={"x-api-token": "owner-secret"}).json()["trace"] is not None
    assert c.post("/check", json={"posting": "x" * 100}).status_code == 200 and seen[-1] == "sample"  # default
    assert c.get("/access").json() == {"owner_resume": False}
    assert c.get("/access", headers={"x-share-token": "share-secret"}).json()["owner_resume"] is True
    assert c.delete("/session/sample").status_code == 400
