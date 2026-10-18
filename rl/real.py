"""Turn your own labeled postings into GroundedMatch episodes, for evaluation only.

Training stays on synthetic data. Your real resume is the exam: it measures whether
what the model learned on made-up resumes carries over to a real one.

    # 1. After `python -m eval.run_eval`, draft labels from Proofline's own answers:
    python -m rl.real draft
    # 2. Open eval/real_labels.jsonl and fix every label by hand. Set "reviewed": true per posting.
    # 3. Score any policy on the reviewed ones:
    python -m rl.evaluate --policy hf:runs/grounded-match --base Qwen/Qwen2.5-1.5B-Instruct --real

The drafts come from the model you're testing against, so they lean toward its mistakes.
Fixing them by hand is the whole point; unreviewed postings are skipped.
eval/real_labels.jsonl contains your resume lines, so it is gitignored and never leaves
your machine.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from proofline import config

from .synth import Episode

LABELS_PATH = Path("eval/real_labels.jsonl")
CACHE_PATH = Path("eval/runs/cache.jsonl")


def resume_lines(path: str = config.BULLETS_PATH) -> list[dict]:
    rows = yaml.safe_load(open(path))
    return [{"id": r["id"], "text": r["text"], "kind": "real", "fam": None}
            for r in rows if r.get("verified") and not r["text"].startswith("<")]


def draft() -> None:
    if not CACHE_PATH.exists():
        sys.exit("run `python -m eval.run_eval` first; drafts are built from its results")
    if LABELS_PATH.exists():
        sys.exit(f"{LABELS_PATH} exists; move it aside before drafting again")
    n = 0
    with LABELS_PATH.open("w") as out:
        for rec in map(json.loads, CACHE_PATH.open()):
            if "report" not in rec:
                continue
            quals = [{"id": f["qual_id"], "text": f["qualification"], "label": f["final"],
                      "valid_ids": [f["bullet_id"]] if f["bullet_id"] else []}
                     for f in rec["report"]["fit"]]
            out.write(json.dumps({"posting_id": rec["id"], "reviewed": False, "quals": quals}) + "\n")
            n += 1
    print(f"wrote {n} drafts to {LABELS_PATH}. For each qualification, check label "
          "(supported / partial / none) and valid_ids (every resume line id that proves it), "
          "then set reviewed: true.")


def load_episodes(path: Path = LABELS_PATH, bullets_path: str = config.BULLETS_PATH) -> list[Episode]:
    lines = resume_lines(bullets_path)
    ids = {ln["id"] for ln in lines}
    eps = []
    for rec in map(json.loads, path.open()):
        if not rec.get("reviewed"):
            continue
        for q in rec["quals"]:
            bad = set(q["valid_ids"]) - ids
            if bad:
                sys.exit(f"{rec['posting_id']} {q['id']}: unknown resume line ids {sorted(bad)}")
            if (q["label"] == "none") != (not q["valid_ids"]):
                sys.exit(f"{rec['posting_id']} {q['id']}: 'none' needs no line ids, other labels need at least one")
        eps.append(Episode(episode_id=f"real-{rec['posting_id']}",
                           quals=[{**q, "fam": None} for q in rec["quals"]],
                           lines=lines, posting_note=None, injected=False))
    if not eps:
        sys.exit(f"no reviewed postings in {path}")
    return eps


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["draft"])
    ap.parse_args()
    draft()
