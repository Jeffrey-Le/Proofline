"""End-to-end smoke test against a running Proofline (local or deployed).

    python -m scripts.smoke --url http://localhost:8000
    python -m scripts.smoke --url https://proofline-xxxx.run.app --share <share-token>

Runs the real pipeline, so it spends a few cents. Checks, in order:
  1. the page and privacy page load
  2. a check against the fictional sample resume streams progress and returns a report
  3. your real resume is refused without the private link (and works with it, if --share)
  4. a visitor resume can be saved, used for a check, and deleted
"""
import argparse
import json
import sys
import time

import httpx

POSTING = """Applied AI Engineering Intern, BS/MS, Summer 2027
Example Cloud Co. - Sunnyvale, CA (in office)

Minimum qualifications:
Currently enrolled in a Bachelor's or Master's program in Computer Science or a related technical field.
Programming experience in Python.
Experience deploying applications on a public cloud platform.

Preferred qualifications:
Experience with vector databases or semantic search.
Must be authorized to work in the US; visa sponsorship is not available."""

RESUME = """Sam Example | sam@example.com | 555-0100
Experience
Software Intern, Acme Corp (2026)
 - Built a Python API that serves 2M requests a day for the billing team
 - Deployed the API to Google Cloud Run with automated tests on every merge
 - Added Redis caching that cut median latency from 180 ms to 40 ms
Research Assistant, State University (2025)
 - Indexed 50k papers with embeddings in a FAISS vector index for semantic search
Education: B.S. Computer Science, State University, May 2027
"""

PROFILE = {"graduation": "2028-05", "degree_level": "BS", "fields": ["Computer Science"],
           "willing_locations": ["Sunnyvale", "Remote"], "open_to_relocation": True,
           "needs_sponsorship": False, "next_program": None}

ok = True
READ_TIMEOUT = 150  # seconds without any data from the server


def step(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail and not cond else ''}")


def stream(c, path, body, headers=None):
    events, result, error = [], None, None
    t0 = time.time()
    try:
      with c.stream("POST", path, json=body, headers=headers or {}) as r:
        if r.status_code != 200:
            return r.status_code, events, None, r.read().decode()[:300]
        for line in r.iter_lines():
            if not line:
                continue
            ev = json.loads(line)
            if "result" in ev:
                result = ev["result"]
            elif "error" in ev:
                error = ev
            else:
                events.append(ev)
                label = ev.get("done") and f"finished {ev['done']}" or ev.get("search") and f"searched: {ev['search']}"
                if label:
                    print(f"         {time.time() - t0:5.1f}s  {label}", flush=True)
    except httpx.ReadTimeout:
        last = next((e for e in reversed(events) if "done" in e), None)
        return 0, events, None, (f"no response for {READ_TIMEOUT}s; last finished step: "
                                 f"{last['done'] if last else 'none'}. Check the server terminal for an ERROR line.")
    return 200, events, result, error


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--share", help="share token, to also test your real resume")
    a = ap.parse_args()
    c = httpx.Client(base_url=a.url.rstrip("/"), timeout=httpx.Timeout(10, read=READ_TIMEOUT))

    print("1. Pages")
    step("home page", c.get("/").status_code == 200)
    step("privacy page", c.get("/privacy").status_code == 200)

    print("2. Sample resume check")
    t = time.time()
    code, events, result, err = stream(c, "/check/stream", {"posting": POSTING, "session_id": "sample"})
    step("streamed a result", code == 200 and result, str(err))
    if result:
        r = result["report"]
        step("progress events arrived before the result", any("steps" in e for e in events))
        step("found eligibility requirements", len(r["eligibility"]) >= 2, json.dumps(r["eligibility"])[:200])
        step("found qualifications", len(r["fit"]) >= 3, json.dumps(r["fit"])[:200])
        step("sample resume proves Python", any("ython" in f["qualification"] and f["final"] != "none" for f in r["fit"]),
             json.dumps([(f["qualification"], f["final"], f["note"]) for f in r["fit"]])[:300])
        step("diagnostics hidden from the public", result["trace"] is None)
        print(f"       verdict: {r['eligible']}, {time.time() - t:.0f}s")

    print("3. Your real resume")
    code, *_ = stream(c, "/check/stream", {"posting": POSTING, "session_id": "owner"})
    step("refused without the private link", code == 403)
    if a.share:
        code, _, result, err = stream(c, "/check/stream", {"posting": POSTING, "session_id": "owner"},
                                      {"x-share-token": a.share})
        step("works with the private link", code == 200 and result, str(err))

    print("4. Visitor resume")
    code, _, saved, err = stream(c, "/session/stream", {"resume": RESUME, "profile": PROFILE})
    step("saved", code == 200 and saved and saved.get("session_id"), str(err))
    if saved and saved.get("session_id"):
        texts = " ".join(b["text"] for b in saved["bullets"])
        step("read the bullet lines", len(saved["bullets"]) >= 3, texts[:200])
        step("did not keep contact details", "sam@example.com" not in texts and "555-0100" not in texts)
        sid = saved["session_id"]
        code, _, result, err = stream(c, "/check/stream", {"posting": POSTING, "session_id": sid})
        step("checked a posting against it", code == 200 and result, str(err))
        if result:
            step("found the Cloud Run line as evidence",
                 any((f["evidence"] or "").find("Cloud Run") >= 0 for f in result["report"]["fit"]),
                 json.dumps([(f["qualification"], f["final"], f["evidence"]) for f in result["report"]["fit"]])[:300])
        step("deleted", c.delete(f"/session/{sid}").status_code == 200)
        code, _, _, err = stream(c, "/check/stream", {"posting": POSTING, "session_id": sid})
        step("gone after deletion", code == 200 and err and err.get("status") == 410, str(err))

    print("\nAll checks passed." if ok else "\nSome checks failed; see FAIL lines above.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
