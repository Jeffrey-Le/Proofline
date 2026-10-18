"""Run the ADK pipeline for one posting and assemble the final report."""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Awaitable, Callable

from google.adk.agents.run_config import RunConfig
from google.adk.runners import InMemoryRunner
from google.genai import types

from . import config, telemetry
from .agents import pipeline, splitter
from .eligibility import grounded, norm
from .store import embed, expiry, get_store
from .telemetry import span
from .schemas import BulletSet, EligibilityFinding, Extraction, FitResult, MatchSet, Report, VerdictSet

APP = "proofline"
logger = logging.getLogger("proofline")
_runner: InMemoryRunner | None = None
_split_runner: InMemoryRunner | None = None

_RANK = {"none": 0, "partial": 1, "supported": 2}


def runner() -> InMemoryRunner:
    global _runner
    if _runner is None:
        _runner = InMemoryRunner(agent=pipeline, app_name=APP)
    return _runner


def split_runner() -> InMemoryRunner:
    global _split_runner
    if _split_runner is None:
        _split_runner = InMemoryRunner(agent=splitter, app_name="proofline_split")
    return _split_runner


def assemble(state: dict) -> Report:
    ex = Extraction.model_validate(state["extraction"])
    elig = [EligibilityFinding(**f) for f in state.get("eligibility", [])]
    matches = {m.qual_id: m for m in MatchSet.model_validate(state.get("matches") or {"matches": []}).matches}
    verdicts = {v.qual_id: v for v in VerdictSet.model_validate(state.get("verdicts") or {"verdicts": []}).verdicts}
    grounding = state.get("grounding", {})

    fit = []
    for q in ex.qualifications:
        m = matches.get(q.id)
        claimed = m.verdict if m else "none"
        final, rejected, note = claimed, None, m.rationale if m else "no match returned"
        if claimed != "none":
            g = grounding.get(q.id, "not checked")
            if g != "ok":
                final, rejected, note = "none", "grounding_check", g
            else:
                v = verdicts.get(q.id)
                if v is None or not v.accepted:
                    final, rejected = "none", "verifier"
                    note = v.reason if v else "this match couldn't be confirmed"
                else:  # the verifier may only downgrade, never upgrade
                    final = min(claimed, v.supported_level, key=_RANK.get)
                    note = v.reason
        keep = final != "none"
        fit.append(FitResult(
            qual_id=q.id, kind=q.kind, qualification=q.text, source_quote=q.source_quote, final=final,
            bullet_id=m.bullet_id if keep else None, evidence=m.evidence_quote if keep else None,
            rejected_by=rejected, note=note,
        ))

    blockers = [f for f in elig if f.status == "blocker"]
    warnings = [f for f in elig if f.status in ("warning", "unknown")]
    eligible = "no" if blockers else ("check" if any(f.kind == "minimum" for f in warnings) else "yes")
    mins = [f for f in fit if f.kind == "minimum"]
    score = sum(_RANK[f.final] / 2 for f in mins) / len(mins) if mins else 0.0
    return Report(company=ex.company, title=ex.title, eligible=eligible,
                  blockers=blockers, warnings=warnings, eligibility=elig, fit=fit,
                  fit_score=round(score, 3))


class SessionExpired(Exception):
    pass


class RunFailed(Exception):
    """A step failed partway (e.g. a model returned JSON that didn't match its schema).
    Carries the state reached so far and which step was running."""

    def __init__(self, state: dict, step: str | None, cause: Exception):
        super().__init__(f"{step or 'pipeline'} failed: {type(cause).__name__}: {str(cause)[:300]}")
        self.state, self.step, self.cause = state, step, cause


Progress = Callable[[dict], Awaitable[None]] | None
STEP_LABELS = {"extractor": "Reading the posting", "matcher": "Searching the resume",
               "formatter": "Organizing matches", "verifier": "Verifying each match",
               "resume_splitter": "Reading the resume"}


async def _run(agent_runner: InMemoryRunner, text: str, state: dict, progress: Progress = None) -> dict:
    """Run one ADK invocation in a throwaway session and return its final state.
    Emits progress events (step started, search made) for the streaming endpoint.
    The ADK session is deleted afterwards so a long-running server doesn't accumulate them."""
    sid = uuid.uuid4().hex
    svc = agent_runner.session_service
    await svc.create_session(app_name=agent_runner.app_name, user_id="u", session_id=sid, state=state)
    try:
        msg = types.Content(role="user", parts=[types.Part(text=text)])
        finished: set[str] = set()
        if progress:
            agents = getattr(agent_runner.agent, "sub_agents", None) or [agent_runner.agent]
            await progress({"steps": [{"id": a.name, "label": STEP_LABELS.get(a.name, a.name)} for a in agents]})
        step = None
        try:
          async for ev in agent_runner.run_async(user_id="u", session_id=sid, new_message=msg,
                                                 run_config=RunConfig(max_llm_calls=config.MAX_LLM_CALLS)):
            if ev.author in STEP_LABELS:
                step = ev.author
            if not progress:
                continue
            if ev.author in STEP_LABELS and ev.is_final_response() and ev.author not in finished:
                finished.add(ev.author)
                await progress({"done": ev.author})
            for fc in ev.get_function_calls() or []:
                if fc.name == "search_resume":
                    await progress({"search": str((fc.args or {}).get("query", ""))[:120]})
        except Exception as e:
            sess = await svc.get_session(app_name=agent_runner.app_name, user_id="u", session_id=sid)
            raise RunFailed(dict(sess.state) if sess else {}, step, e) from e
        sess = await svc.get_session(app_name=agent_runner.app_name, user_id="u", session_id=sid)
        return dict(sess.state)
    finally:
        await svc.delete_session(app_name=agent_runner.app_name, user_id="u", session_id=sid)


async def check_posting(posting: str, session_id: str = config.OWNER_SESSION,
                        cloud_trace: str | None = None, progress: Progress = None) -> tuple[Report, dict]:
    """Check a posting against the owner's resume, the fictional sample, or a visitor's session."""
    tr = telemetry.Trace(cloud_trace=cloud_trace)
    token = telemetry.current.set(tr)
    try:
        state = {"posting": posting, "session_id": session_id, "evidence_block": "", "claims_block": ""}
        profile = get_store().get_profile(session_id)
        if profile is not None:
            state["profile"] = profile
        elif session_id != config.OWNER_SESSION:  # owner falls back to data/profile.yaml for local runs
            raise SessionExpired()
        text, notes = f"<posting>\n{posting}\n</posting>", []
        try:
            final = await _run(runner(), text, dict(state), progress)
        except RunFailed as first:
            logger.warning("check attempt 1 failed in %s; retrying: %s", first.step, first)
            try:
                final = await _run(runner(), text, dict(state), progress)
            except RunFailed as second:
                logger.error("check attempt 2 failed in %s: %s", second.step, second, exc_info=second.cause)
                # Only the last step failed: keep everything before it. Unverified matches are
                # shown as not counted, never as proven (the verifier may only downgrade).
                if "matches" in second.state and "extraction" in second.state:
                    final = {**second.state, "verdicts": None}
                    notes.append("Some matches couldn't be double-checked this time, so they aren't counted. "
                                 "Run the check again to try verifying them.")
                else:
                    raise second.cause
        report = assemble(final)
        report.notes = notes
        kind = session_id if session_id in (config.OWNER_SESSION, config.SAMPLE_SESSION) else "visitor"
        log = telemetry.emit(tr, {"op": "check", "resume": kind,
                                  "eligible": report.eligible, "n_blockers": len(report.blockers),
                                  "n_searches": len(final.get("searches") or [])})
        return report, log
    finally:
        telemetry.current.reset(token)


async def create_session(resume: str, profile: dict, progress: Progress = None) -> tuple[dict, dict]:
    """Split a visitor's resume into verbatim bullets, embed them and store them for 24 hours.
    Only bullets found word for word in the resume are kept; the resume text itself is never stored."""
    tr = telemetry.Trace()
    token = telemetry.current.set(tr)
    try:
        final = await _run(split_runner(), f"<resume>\n{resume}\n</resume>", {}, progress)
        proposed = BulletSet.model_validate(final.get("bullets") or {"bullets": []}).bullets
        with span("bullet_grounding", "rule", proposed=len(proposed)):
            seen, kept = set(), []
            for b in proposed:
                key = norm(b.text)
                if len(key) >= 25 and key not in seen and grounded(b.text, resume):
                    seen.add(key)
                    kept.append(b)
            kept = kept[:config.MAX_BULLETS]
        sid = uuid.uuid4().hex
        expire_at = expiry()
        if kept:
            vecs = embed([b.text for b in kept], "RETRIEVAL_DOCUMENT")
            rows = [{"id": f"b{i+1:02d}", "text": b.text, "source": b.source[:120], "vector": v}
                    for i, (b, v) in enumerate(zip(kept, vecs))]
            with span("store_session", "retrieval", n=len(rows)):
                get_store().put_session(sid, rows, profile, expire_at)
        out = {"session_id": sid if kept else None, "expires_at": expire_at.isoformat(),
               "bullets": [{"id": f"b{i+1:02d}", "text": b.text, "source": b.source} for i, b in enumerate(kept)],
               "dropped": len(proposed) - len(kept)}
        log = telemetry.emit(tr, {"op": "create_session", "n_bullets": len(kept), "n_dropped": out["dropped"]})
        return out, log
    finally:
        telemetry.current.reset(token)


def delete_session(session_id: str) -> None:
    if session_id in (config.OWNER_SESSION, config.SAMPLE_SESSION):
        raise ValueError("cannot delete built-in resumes")
    get_store().delete_session(session_id)


def check_posting_sync(posting: str) -> tuple[Report, dict]:
    return asyncio.run(check_posting(posting))
