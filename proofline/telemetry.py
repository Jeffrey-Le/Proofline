"""Per-request trace: one span per step with latency, tokens and cost.

A ContextVar holds the current request's Trace, so ADK model callbacks (which
don't know about our request) can still append to it. Each request ends with a
single JSON log line; on Cloud Run, Cloud Logging parses it and links it to the
request's Cloud Trace via the `logging.googleapis.com/trace` field.
"""
from __future__ import annotations

import contextvars
import json
import os
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Optional

from .config import PRICES


@dataclass
class Span:
    name: str
    kind: str  # llm | embed | retrieval | rule
    start_ms: float
    latency_ms: float = 0.0
    model: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cost_usd: Optional[float] = None
    attrs: dict = field(default_factory=dict)


@dataclass
class Trace:
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    cloud_trace: Optional[str] = None
    t0: float = field(default_factory=time.perf_counter)
    spans: list[Span] = field(default_factory=list)
    _open: dict = field(default_factory=dict)

    def now_ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000

    def summary(self) -> dict:
        costs = [s.cost_usd for s in self.spans if s.cost_usd is not None]
        return {
            "request_id": self.request_id,
            "total_latency_ms": round(self.now_ms(), 1),
            "llm_calls": sum(s.kind == "llm" for s in self.spans),
            "input_tokens": sum(s.input_tokens for s in self.spans),
            "output_tokens": sum(s.output_tokens + s.thinking_tokens for s in self.spans),
            "cost_usd": round(sum(costs), 6) if costs else None,
            "spans": [{k: v for k, v in asdict(s).items() if v not in (None, {}, 0, 0.0)} | {"name": s.name}
                      for s in self.spans],
        }


current: contextvars.ContextVar[Optional[Trace]] = contextvars.ContextVar("trace", default=None)


def price(model: Optional[str], tin: int, tout: int) -> Optional[float]:
    p = PRICES.get(model or "")
    return None if p is None else (tin * p[0] + tout * p[1]) / 1e6


@contextmanager
def span(name: str, kind: str, **attrs):
    tr = current.get()
    if tr is None:
        yield None
        return
    s = Span(name=name, kind=kind, start_ms=round(tr.now_ms(), 1), attrs=attrs)
    t = time.perf_counter()
    try:
        yield s
    finally:
        s.latency_ms = round((time.perf_counter() - t) * 1000, 1)
        tr.spans.append(s)


# ---------- ADK callbacks ----------
def before_model(callback_context, llm_request):
    tr = current.get()
    if tr is not None:
        tr._open[callback_context.agent_name] = (time.perf_counter(), tr.now_ms(), llm_request.model)
    return None


def after_model(callback_context, llm_response):
    tr = current.get()
    if tr is None or llm_response.partial:
        return None
    t, start_ms, model = tr._open.pop(callback_context.agent_name, (time.perf_counter(), tr.now_ms(), None))
    u = llm_response.usage_metadata
    tin = (u.prompt_token_count or 0) if u else 0
    tout = (u.candidates_token_count or 0) if u else 0
    think = (getattr(u, "thoughts_token_count", 0) or 0) if u else 0
    tr.spans.append(Span(
        name=callback_context.agent_name, kind="llm", start_ms=round(start_ms, 1),
        latency_ms=round((time.perf_counter() - t) * 1000, 1), model=model,
        input_tokens=tin, output_tokens=tout, thinking_tokens=think,
        cost_usd=price(model, tin, tout + think),
    ))
    return None


def emit(tr: Trace, extra: dict | None = None) -> dict:
    rec = {"severity": "INFO", "message": "proofline.request", **tr.summary(), **(extra or {})}
    if tr.cloud_trace and (proj := os.getenv("GOOGLE_CLOUD_PROJECT")):
        rec["logging.googleapis.com/trace"] = f"projects/{proj}/traces/{tr.cloud_trace}"
    print(json.dumps(rec, default=str), file=sys.stdout, flush=True)
    return rec
