"""Evaluate blocker detection on your labeled postings.

    python -m eval.run_eval --labels eval/labels.jsonl [--sleep 4] [--fresh]

labels.jsonl, one object per line:
  {"id": "google-fde-2027", "posting_path": "eval/postings/google-fde-2027.txt",
   "expected_blockers": ["return_to_school"], "expected_flags": [], "expected_eligible": "no"}

expected_blockers  hard gates you'd personally rule as disqualifying (category names from schemas.Category)
expected_flags     soft issues you'd want surfaced (preferred-only, depends-on-admission, etc.)
expected_eligible  yes | no | check

Results are cached by posting hash in eval/runs/cache.jsonl so re-running the scorer
doesn't spend free-tier quota. Use --fresh after changing prompts.
"""
import argparse
import asyncio
import hashlib
import json
import statistics as st
import time
from pathlib import Path

from proofline.pipeline import check_posting

RUNS = Path(__file__).parent / "runs"


def load_cache(fresh: bool) -> dict:
    f = RUNS / "cache.jsonl"
    if fresh or not f.exists():
        return {}
    return {(r := json.loads(l))["key"]: r for l in f.open()}


async def run_all(labels, sleep, fresh):
    RUNS.mkdir(exist_ok=True)
    cache, out = load_cache(fresh), []
    with (RUNS / "cache.jsonl").open("a" if not fresh else "w") as fh:
        for i, lab in enumerate(labels):
            text = Path(lab["posting_path"]).read_text()
            key = hashlib.sha256(text.encode()).hexdigest()[:16]
            if key in cache and "error" not in cache[key]:  # errors are retried, not reused
                out.append((lab, cache[key]))
                continue
            try:
                report, trace = await check_posting(text)
                rec = {"key": key, "id": lab["id"], "report": report.model_dump(), "trace": trace}
            except Exception as e:  # keep going; one bad posting shouldn't kill the run
                rec = {"key": key, "id": lab["id"], "error": repr(e)}
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            out.append((lab, rec))
            print(f"[{i+1}/{len(labels)}] {lab['id']}: {rec.get('report', {}).get('eligible', rec.get('error'))}")
            time.sleep(sleep)  # respect free-tier RPM
    return out


def score(results):
    tp = fp = fn = flag_tp = flag_fn = 0
    missed, false_block, verdict_ok, errors = [], [], 0, 0
    lat, tok, cost = [], [], []
    claims = {"proposed": 0, "grounding_rejected": 0, "verifier_rejected": 0}
    for lab, rec in results:
        if "error" in rec:
            errors += 1
            continue
        rep, tr = rec["report"], rec["trace"]
        pred_b = {b["category"] for b in rep["blockers"]}
        pred_any = pred_b | {w["category"] for w in rep["warnings"]}
        exp_b, exp_f = set(lab.get("expected_blockers", [])), set(lab.get("expected_flags", []))
        tp += len(pred_b & exp_b); fp += len(pred_b - exp_b); fn += len(exp_b - pred_b)
        flag_tp += len(exp_f & pred_any); flag_fn += len(exp_f - pred_any)
        if exp_b - pred_b:
            missed.append((lab["id"], sorted(exp_b - pred_b)))
        if pred_b - exp_b:
            false_block.append((lab["id"], sorted(pred_b - exp_b)))
        verdict_ok += rep["eligible"] == lab.get("expected_eligible")
        lat.append(tr["total_latency_ms"]); tok.append(tr["input_tokens"] + tr["output_tokens"])
        if tr.get("cost_usd") is not None:
            cost.append(tr["cost_usd"])
        for f in rep["fit"]:
            claims["proposed"] += f["final"] != "none" or f["rejected_by"] is not None
            claims["grounding_rejected"] += f["rejected_by"] == "grounding_check"
            claims["verifier_rejected"] += f["rejected_by"] == "verifier"

    n = len(results) - errors
    pct = lambda a, b: f"{100*a/b:.1f}%" if b else "n/a"
    q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))] if xs else None
    return {
        "postings_scored": n, "errors": errors,
        "blocker_recall": pct(tp, tp + fn), "blocker_precision": pct(tp, tp + fp),
        "postings_with_missed_blocker": len(missed), "missed": missed,
        "postings_with_false_blocker": len(false_block), "false_blocks": false_block,
        "soft_flag_recall": pct(flag_tp, flag_tp + flag_fn),
        "eligible_verdict_accuracy": pct(verdict_ok, n),
        "latency_ms_p50": q(lat, .5), "latency_ms_p95": q(lat, .95),
        "tokens_per_request_mean": round(st.mean(tok)) if tok else None,
        "cost_per_request_usd_mean": round(st.mean(cost), 6) if cost else None,
        "fit_claims": claims,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="eval/labels.jsonl")
    ap.add_argument("--sleep", type=float, default=4.0)
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    labels = [json.loads(l) for l in open(a.labels) if l.strip()]
    results = asyncio.run(run_all(labels, a.sleep, a.fresh))
    s = score(results)
    (RUNS / "summary.json").write_text(json.dumps(s, indent=2))
    (RUNS / "results.md").write_text(markdown(s, results))
    print(json.dumps(s, indent=2))
    print(f"\nREADME table and misses written to {RUNS / 'results.md'}")


def markdown(s: dict, results) -> str:
    """The README Results table plus the misses to write a failure analysis from."""
    sec = lambda ms: f"{ms / 1000:.1f} s" if ms is not None else "n/a"
    c = s["fit_claims"]
    has_block = sum(1 for lab, _ in results if lab.get("expected_blockers"))
    caught = has_block - s["postings_with_missed_blocker"]
    eligible = sum(1 for lab, rec in results if not lab.get("expected_blockers") and "error" not in rec)
    cost = s["cost_per_request_usd_mean"]
    rows = [
        ("Postings scored", f"{s['postings_scored']} ({s['errors']} errors)"),
        ("Blocker recall (postings with a real blocker that was caught)", f"{caught}/{has_block} postings · {s['blocker_recall']} of blockers"),
        ("Blocker precision", s["blocker_precision"]),
        ("False blocks (eligible postings marked not eligible)", f"{s['postings_with_false_blocker']}/{eligible}"),
        ("Soft-flag recall", s["soft_flag_recall"]),
        ("Overall verdict accuracy", s["eligible_verdict_accuracy"]),
        ("Fit claims rejected: grounding gate / verifier", f"{c['grounding_rejected']} / {c['verifier_rejected']} of {c['proposed']} proposed"),
        ("Latency p50 / p95", f"{sec(s['latency_ms_p50'])} / {sec(s['latency_ms_p95'])}"),
        ("Cost per check (measured)", f"${cost:.4f}" if cost is not None else "n/a"),
    ]
    miss = [f"- missed blocker on `{pid}`: {', '.join(cats)}" for pid, cats in s["missed"]]
    miss += [f"- false block on `{pid}`: {', '.join(cats)}" for pid, cats in s["false_blocks"]]
    miss += [f"- verdict on `{lab['id']}`: said {rec['report']['eligible']}, label {lab.get('expected_eligible')}"
             for lab, rec in results if "report" in rec and rec["report"]["eligible"] != lab.get("expected_eligible")]
    miss += [f"- error on `{lab['id']}`: {rec['error'][:200]}" for lab, rec in results if "error" in rec]
    out = ["| Metric | Value |", "|---|---|", *[f"| {k} | {v} |" for k, v in rows], "",
           "## Misses (for the failure analysis)", "", *(miss or ["- none"])]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
