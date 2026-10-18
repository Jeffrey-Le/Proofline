"""Embeddings, vector store and per-session profiles.

Every bullet row carries a session_id: "owner" for your own resume, a random hex id
for each visitor. Searches always filter by session_id, so one visitor can never
retrieve another's resume. Visitor rows carry expire_at; Firestore's TTL policy deletes
them, and reads also check expiry in code because TTL deletion can lag by a day.

Two backends with the same interface: LanceDB locally, Firestore when deployed.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional, Protocol

from . import config
from .telemetry import price, span

_SID = re.compile(r"^(owner|sample|[0-9a-f]{32})$")


def check_sid(sid: str) -> str:
    if not _SID.match(sid or ""):
        raise ValueError("bad session id")
    return sid


def now() -> datetime:
    return datetime.now(timezone.utc)


def expiry() -> datetime:
    return now() + timedelta(hours=config.SESSION_TTL_HOURS)


@dataclass
class Hit:
    id: str
    text: str
    source: str
    score: float  # cosine similarity, higher is better


# ---------- embeddings ----------
@lru_cache
def _client():
    from google import genai
    return genai.Client()  # reads GEMINI_API_KEY


def _l2(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]  # truncated-dim Gemini embeddings must be re-normalized


def embed(texts: list[str], task: str) -> list[list[float]]:
    """task: RETRIEVAL_DOCUMENT for bullets, RETRIEVAL_QUERY for qualifications."""
    from google.genai import types
    with span("embed", "embed", model=config.EMBED_MODEL, n=len(texts), task=task) as s:
        out = []
        for i in range(0, len(texts), 100):  # API batch limit
            r = _client().models.embed_content(
                model=config.EMBED_MODEL, contents=texts[i:i + 100],
                config=types.EmbedContentConfig(task_type=task, output_dimensionality=config.EMBED_DIM),
            )
            out += [_l2(e.values) for e in r.embeddings]
        if s is not None:  # embedding responses don't report tokens; ~4 chars per token
            s.model, s.input_tokens = config.EMBED_MODEL, sum(len(t) for t in texts) // 4
            s.cost_usd = price(config.EMBED_MODEL, s.input_tokens, 0)
        return out


# ---------- stores ----------
class VectorStore(Protocol):
    def put_session(self, sid: str, rows: list[dict], profile: Optional[dict], expire_at: Optional[datetime]) -> None: ...
    def search(self, vec: list[float], k: int, sid: str) -> list[Hit]: ...
    def get_profile(self, sid: str) -> Optional[dict]: ...
    def delete_session(self, sid: str) -> None: ...


class LanceStore:
    """Local dev backend. Profiles live in small JSON files next to the table."""

    def __init__(self, path: str = config.LANCEDB_PATH, table: str = "bullets"):
        import lancedb
        self.path, self.db, self.name = Path(path), lancedb.connect(path), table
        (self.path / "sessions").mkdir(parents=True, exist_ok=True)

    def _table(self):
        return self.db.open_table(self.name) if self.name in self.db.table_names() else None

    def put_session(self, sid, rows, profile, expire_at):
        check_sid(sid)
        exp = expire_at.isoformat() if expire_at else ""
        data = [{"id": r["id"], "text": r["text"], "source": r["source"], "vector": r["vector"],
                 "session_id": sid, "expire_at": exp} for r in rows]
        t = self._table()
        if t is None:
            self.db.create_table(self.name, data=data)
        else:
            t.delete(f"session_id = '{sid}'")
            t.add(data)
        if profile is not None:
            (self.path / "sessions" / f"{sid}.json").write_text(json.dumps({"profile": profile, "expire_at": exp}))

    def _expired(self, exp: str) -> bool:
        return bool(exp) and datetime.fromisoformat(exp) < now()

    def search(self, vec, k, sid):
        check_sid(sid)
        t = self._table()
        if t is None:
            return []
        res = (t.search(vec).metric("cosine").where(f"session_id = '{sid}'", prefilter=True)
               .limit(k).to_list())
        return [Hit(r["id"], r["text"], r["source"], 1 - r["_distance"]) for r in res
                if not self._expired(r["expire_at"])]

    def get_profile(self, sid):
        f = self.path / "sessions" / f"{check_sid(sid)}.json"
        if not f.exists():
            return None
        d = json.loads(f.read_text())
        return None if self._expired(d["expire_at"]) else d["profile"]

    def delete_session(self, sid):
        check_sid(sid)
        if (t := self._table()) is not None:
            t.delete(f"session_id = '{sid}'")
        (self.path / "sessions" / f"{sid}.json").unlink(missing_ok=True)


class FirestoreStore:
    """Indexes and TTL policies are created by deploy.sh."""

    def __init__(self):
        from google.cloud import firestore
        self.db = firestore.Client()
        self.col = self.db.collection(config.FIRESTORE_COLLECTION)
        self.sessions = self.db.collection(config.SESSIONS_COLLECTION)

    def put_session(self, sid, rows, profile, expire_at):
        from google.cloud.firestore_v1.vector import Vector
        check_sid(sid)
        self._delete_bullets(sid)
        batch = self.db.batch()
        for r in rows:
            doc = {"id": r["id"], "text": r["text"], "source": r["source"],
                   "vector": Vector(r["vector"]), "session_id": sid}
            if expire_at:
                doc["expire_at"] = expire_at
            batch.set(self.col.document(f"{sid}_{r['id']}"), doc)
        if profile is not None:
            s = {"profile": profile}
            if expire_at:
                s["expire_at"] = expire_at
            batch.set(self.sessions.document(sid), s)
        batch.commit()

    def search(self, vec, k, sid):
        from google.cloud.firestore_v1.base_query import FieldFilter
        from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
        from google.cloud.firestore_v1.vector import Vector
        q = (self.col.where(filter=FieldFilter("session_id", "==", check_sid(sid)))
             .find_nearest(vector_field="vector", query_vector=Vector(vec), limit=k,
                           distance_measure=DistanceMeasure.COSINE, distance_result_field="dist"))
        out = []
        for d in q.stream():
            exp = d.get("expire_at") if "expire_at" in (d.to_dict() or {}) else None
            if exp is None or exp > now():
                out.append(Hit(d.get("id"), d.get("text"), d.get("source"), 1 - d.get("dist")))
        return out

    def get_profile(self, sid):
        d = self.sessions.document(check_sid(sid)).get()
        if not d.exists:
            return None
        data = d.to_dict()
        if data.get("expire_at") and data["expire_at"] < now():
            return None
        return data["profile"]

    def _delete_bullets(self, sid):
        from google.cloud.firestore_v1.base_query import FieldFilter
        for d in self.col.where(filter=FieldFilter("session_id", "==", sid)).stream():
            d.reference.delete()

    def delete_session(self, sid):
        check_sid(sid)
        self._delete_bullets(sid)
        self.sessions.document(sid).delete()


@lru_cache
def get_store() -> VectorStore:
    return FirestoreStore() if config.VECTOR_BACKEND == "firestore" else LanceStore()


def retrieve(queries: list[str], sid: str, k: int = config.TOP_K) -> list[list[Hit]]:
    vecs = embed(queries, "RETRIEVAL_QUERY")
    with span("vector_search", "retrieval", backend=config.VECTOR_BACKEND, n=len(queries), k=k):
        return [get_store().search(v, k, sid) for v in vecs]
