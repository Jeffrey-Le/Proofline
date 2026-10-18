"""HTTP front end for Cloud Run.

POST   /check             {"posting": "...", "session_id": optional}  -> report + trace
POST   /check/stream      same, as NDJSON: progress events, then {"result": ...}
POST   /session           {"resume": "...", "profile": {...}}         -> visitor session (24 h)
DELETE /session/{id}                                                   -> delete it now
GET    /                  the page

Limits for public traffic (your owner token skips them):
  - per IP: PER_IP_PER_HOUR requests
  - per day: DAILY_BUDGET_USD of measured model spend; when it's used up the public
    demo pauses until midnight Pacific. This is the real cost cap; the request
    counters just stop one visitor from using the whole budget.
With max-instances=1 these in-memory counters are exact.
"""
import asyncio
import json
import logging
import os
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from proofline import config
from proofline.pipeline import SessionExpired, check_posting, create_session, delete_session
from proofline.schemas import ProfileIn
from proofline.store import check_sid

app = FastAPI(title="proofline")
MAX_CHARS = 20_000
API_TOKEN = os.getenv("API_TOKEN")      # you: skips limits, sees diagnostics
SHARE_TOKEN = os.getenv("SHARE_TOKEN")  # recruiters: may check against your real resume, normal limits
# Shown only to owner/share-link holders, never in public HTML. Set via env so the code names nobody.
OWNER_LABEL = os.getenv("OWNER_LABEL", "Shared resume")
OWNER_NOTE = os.getenv("OWNER_NOTE", "A resume shared with you through a private link.")
PER_IP_PER_HOUR = int(os.getenv("PER_IP_PER_HOUR", "8"))
PAGE = Path(__file__).parent / "static" / "index.html"
PT = ZoneInfo("America/Los_Angeles")

_ip_hits: dict[str, deque] = defaultdict(deque)
_spend = {"date": None, "usd": 0.0, "requests": 0}


class CheckIn(BaseModel):
    posting: str
    session_id: str | None = None


class SessionIn(BaseModel):
    resume: str = Field(max_length=config.MAX_RESUME_CHARS)
    profile: ProfileIn


def _today():
    d = datetime.now(PT).date()
    if _spend["date"] != d:
        _spend.update(date=d, usd=0.0, requests=0)


def _gate(request: Request, token: str | None, xff: str | None) -> bool:
    """Raise 429 if a public caller is over a limit. Returns True for the owner."""
    _today()
    if API_TOKEN and token == API_TOKEN:
        return True
    if _spend["usd"] >= config.DAILY_BUDGET_USD:
        raise HTTPException(429, "Proofline has reached its daily limit. Please try again tomorrow.")
    ip = (xff or "").split(",")[0].strip() or (request.client.host if request.client else "?")
    now = time.time()
    for k in [k for k, v in _ip_hits.items() if not v or now - v[-1] > 3600]:
        del _ip_hits[k]  # forget IPs once their hour has passed
    q = _ip_hits[ip]
    while q and now - q[0] > 3600:
        q.popleft()
    if len(q) >= PER_IP_PER_HOUR:
        raise HTTPException(429, f"You've reached the limit of {PER_IP_PER_HOUR} checks an hour. Please try again a bit later.")
    q.append(now)
    return False


def _charge(log: dict):
    _spend["usd"] += log.get("cost_usd") or 0.0
    _spend["requests"] += 1


log = logging.getLogger("proofline")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def _model_error(e: Exception) -> HTTPException:
    # Full error in the server log (never shown to users); a plain message for the page.
    log.error("check failed: %s: %s", type(e).__name__, str(e)[:500], exc_info=e)
    if getattr(e, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(e):
        return HTTPException(503, "Proofline is busy right now. Please try again in a minute.")
    return HTTPException(502, "Something went wrong during the check. Please try again.")


@app.get("/access")
def access(x_api_token: str | None = Header(default=None), x_share_token: str | None = Header(default=None)):
    """Lets the page know whether to offer the private shared-resume option, and what to call it."""
    if not _can_use_owner(x_api_token, x_share_token):
        return {"owner_resume": False}
    return {"owner_resume": True, "label": OWNER_LABEL, "note": OWNER_NOTE}


@app.get("/healthz")
def health():
    _today()
    return {"ok": True, "spent_today_usd": round(_spend["usd"], 4), "requests_today": _spend["requests"]}


def _can_use_owner(token: str | None, share: str | None) -> bool:
    return bool(API_TOKEN and token == API_TOKEN) or bool(SHARE_TOKEN and share == SHARE_TOKEN)


def _prepare_check(body: CheckIn, request: Request, token, xff, share=None) -> tuple[str, str, bool]:
    text = body.posting.strip()
    if not 50 < len(text) < MAX_CHARS:
        raise HTTPException(400, f"Paste the full posting (50 to {MAX_CHARS:,} characters).")
    sid = body.session_id or config.SAMPLE_SESSION
    try:
        check_sid(sid)
    except ValueError:
        raise HTTPException(400, "Unknown resume session. Save your resume again.")
    if sid == config.OWNER_SESSION and not _can_use_owner(token, share):
        raise HTTPException(403, "This resume is only available through a private link.")
    owner = _gate(request, token, xff)
    return text, sid, owner


async def _do_check(text: str, sid: str, trace_id: str | None, owner: bool, progress=None) -> dict:
    """The trace (timings, tokens, cost) is logged for every request but only returned to the owner."""
    try:
        report, log = await check_posting(text, session_id=sid, cloud_trace=trace_id, progress=progress)
    except SessionExpired:
        raise HTTPException(410, "Your resume was deleted after 24 hours. Add it again to keep checking.")
    except Exception as e:
        raise _model_error(e)
    _charge(log)
    return {"report": report.model_dump(), "trace": log if owner else None}


def _prepare_session(body: SessionIn, request: Request, token, xff) -> None:
    if len(body.resume.strip()) < 200:
        raise HTTPException(400, "Paste your full resume text (at least a few bullets).")
    _gate(request, token, xff)


async def _do_session(body: SessionIn, progress=None) -> dict:
    try:
        out, log = await create_session(body.resume, body.profile.model_dump(), progress=progress)
    except Exception as e:
        raise _model_error(e)
    _charge(log)
    if not out["session_id"]:
        raise HTTPException(422, "No resume bullets could be read from that text. Paste the plain text of your resume, including bullet points.")
    return out


def _stream(work) -> StreamingResponse:
    """Run `work(progress)` and stream newline-delimited JSON: progress events while it runs,
    then {"result": ...} or {"error": ..., "status": ...}."""
    q: asyncio.Queue = asyncio.Queue()

    async def progress(event: dict):
        await q.put(event)

    async def runner():
        try:
            await q.put({"result": await work(progress)})
        except HTTPException as e:
            await q.put({"error": e.detail, "status": e.status_code})
        except Exception as e:
            log.error("stream failed: %s: %s", type(e).__name__, str(e)[:500], exc_info=e)
            await q.put({"error": "Something went wrong on our side. Please try again.", "status": 500})
        finally:
            await q.put(None)

    task = asyncio.create_task(runner())

    async def body():
        while (item := await q.get()) is not None:
            yield json.dumps(item) + "\n"
        await task

    return StreamingResponse(body(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


def _trace_id(header: str | None) -> str | None:
    return (header or "").split("/")[0] or None


@app.post("/check")
async def check(body: CheckIn, request: Request, x_api_token: str | None = Header(default=None),
                x_share_token: str | None = Header(default=None),
                x_cloud_trace_context: str | None = Header(default=None),
                x_forwarded_for: str | None = Header(default=None)):
    text, sid, owner = _prepare_check(body, request, x_api_token, x_forwarded_for, x_share_token)
    return await _do_check(text, sid, _trace_id(x_cloud_trace_context), owner)


@app.post("/check/stream")
async def check_stream(body: CheckIn, request: Request, x_api_token: str | None = Header(default=None),
                       x_share_token: str | None = Header(default=None),
                       x_cloud_trace_context: str | None = Header(default=None),
                       x_forwarded_for: str | None = Header(default=None)):
    text, sid, owner = _prepare_check(body, request, x_api_token, x_forwarded_for, x_share_token)
    return _stream(lambda progress: _do_check(text, sid, _trace_id(x_cloud_trace_context), owner, progress))


@app.post("/session")
async def new_session(body: SessionIn, request: Request, x_api_token: str | None = Header(default=None),
                      x_forwarded_for: str | None = Header(default=None)):
    _prepare_session(body, request, x_api_token, x_forwarded_for)
    return await _do_session(body)


@app.post("/session/stream")
async def new_session_stream(body: SessionIn, request: Request, x_api_token: str | None = Header(default=None),
                             x_forwarded_for: str | None = Header(default=None)):
    _prepare_session(body, request, x_api_token, x_forwarded_for)
    return _stream(lambda progress: _do_session(body, progress))


@app.delete("/session/{sid}")
def remove_session(sid: str):
    try:
        check_sid(sid)
        delete_session(sid)
    except ValueError:
        raise HTTPException(400, "Unknown session.")
    return {"deleted": True}


@app.get("/")
def index():
    return FileResponse(PAGE)


@app.get("/privacy")
def privacy():
    return FileResponse(PAGE.parent / "privacy.html")
