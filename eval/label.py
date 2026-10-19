"""Save and label postings for the eval, blind (Proofline's answer is never shown here).

    python -m eval.label add google-fde-2027        # paste a posting, then Ctrl-D (or --clipboard on a Mac)
    python -m eval.label                            # label every posting that has no label yet
    python -m eval.label redo google-fde-2027       # label one posting again
    python -m eval.label status                     # how balanced your set is

Labels go to eval/labels.jsonl. Posting text stays in eval/postings/ (gitignored).
Label as yourself: a blocker is something you'd personally accept as "I can't apply",
a flag is something you'd want pointed out but that doesn't rule you out.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import yaml

from proofline import config

POSTINGS = Path("eval/postings")
LABELS = Path("eval/labels.jsonl")

CATEGORIES = [
    ("graduation_window", "graduation date outside the allowed window"),
    ("enrollment", "must be currently enrolled during the program"),
    ("return_to_school", "must return to school after the internship"),
    ("degree_level", "requires a degree level you don't have (MS, PhD)"),
    ("degree_field", "requires a field you aren't in"),
    ("location", "office or location you can't do"),
    ("work_authorization", "sponsorship or work authorization"),
    ("program_dates", "program dates you can't make"),
    ("other", "any other hard requirement"),
]
NAMES = [c for c, _ in CATEGORIES]

# Lines worth reading first: where eligibility rules usually hide.
KEY = re.compile(
    r"graduat|degree|bachelor|master|ph\.?d|\bbs\b|\bms\b|enroll|currently pursuing|return(ing)? to|"
    r"penultimate|final year|class of|20\d\d|sponsor|visa|authoriz|citizen|clearance|relocat|on-?site|"
    r"hybrid|remote|location|office|minimum qualif|basic qualif|required|must|start date|weeks|summer",
    re.I,
)
SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{1,60}$")


def key_lines(text: str, limit: int = 25) -> list[str]:
    out = []
    for ln in (l.strip() for l in text.splitlines()):
        if ln and KEY.search(ln) and ln not in out:
            out.append(ln if len(ln) <= 220 else ln[:217] + "...")
        if len(out) >= limit:
            break
    return out


def parse_choice(raw: str, n: int = len(NAMES)) -> list[str]:
    """'1,3' or '1 3' -> category names. Blank -> []. Raises ValueError on anything else."""
    raw = raw.strip()
    if not raw:
        return []
    picks = []
    for tok in re.split(r"[,\s]+", raw):
        if not tok.isdigit() or not 1 <= int(tok) <= n:
            raise ValueError(tok)
        name = NAMES[int(tok) - 1]
        if name not in picks:
            picks.append(name)
    return picks


def default_verdict(blockers: list[str], flags: list[str]) -> str:
    return "no" if blockers else "check" if flags else "yes"


def read_labels() -> dict[str, dict]:
    if not LABELS.exists():
        return {}
    return {(r := json.loads(l))["id"]: r for l in LABELS.open() if l.strip()}


def write_labels(labels: dict[str, dict]) -> None:
    LABELS.write_text("".join(json.dumps(r) + "\n" for r in labels.values()))


def profile_summary() -> str:
    path = Path(config.PROFILE_PATH)
    if not path.exists():
        path = Path("data/profile.example.yaml")
    p = yaml.safe_load(path.read_text())
    nxt = p.get("next_program")
    parts = [f"graduates {p['graduation']}", p["degree_level"], ", ".join(p["fields"]),
             "needs sponsorship" if p.get("needs_sponsorship") else "no sponsorship needed",
             "locations: " + ", ".join(p["willing_locations"])]
    if nxt:
        parts.append(f"next: {nxt.get('level')} from {nxt.get('start')} ({nxt.get('status')})")
    return " · ".join(parts)


def ask(prompt: str, parse, default=None):
    while True:
        raw = input(prompt)
        if not raw.strip() and default is not None:
            return default
        try:
            return parse(raw)
        except ValueError as e:
            print(f"  didn't understand {e}; try again")


def label_one(pid: str, text: str) -> dict | None:
    print("\n" + "=" * 78 + f"\n{pid}   ({len(text):,} characters)\n" + "=" * 78)
    for ln in key_lines(text):
        print("  · " + ln)
    print("\n(type 'f' to see the full posting, 's' to skip, 'q' to quit)")
    for i, (c, desc) in enumerate(CATEGORIES, 1):
        print(f"  {i}. {c:<19} {desc}")

    def choice(raw):
        r = raw.strip().lower()
        if r in {"f", "s", "q"}:
            return r
        return parse_choice(raw)

    while True:
        b = ask("\nBlockers that rule you out (numbers, blank for none): ", choice, default=[])
        if b == "f":
            print("\n" + text + "\n")
            continue
        break
    if b in ("s", "q"):
        return None if b == "s" else "quit"
    f = ask("Flags you'd want pointed out (numbers, blank for none): ", parse_choice, default=[])
    f = [x for x in f if x not in b]
    d = default_verdict(b, f)

    def verdict(raw):
        r = raw.strip().lower()
        r = {"y": "yes", "n": "no", "c": "check"}.get(r, r)
        if r not in {"yes", "no", "check"}:
            raise ValueError(raw)
        return r

    e = ask(f"Overall: can you apply? yes / no / check [{d}]: ", verdict, default=d)
    note = input("Note for later (optional): ").strip()
    return {"id": pid, "posting_path": str(POSTINGS / f"{pid}.txt"), "expected_blockers": b,
            "expected_flags": f, "expected_eligible": e, "notes": note}


def cmd_add(pid: str, clipboard: bool) -> None:
    if not SLUG.match(pid):
        sys.exit("use a short id like google-fde-2027 (lowercase letters, digits, dashes)")
    POSTINGS.mkdir(parents=True, exist_ok=True)
    path = POSTINGS / f"{pid}.txt"
    if path.exists():
        sys.exit(f"{path} already exists")
    if clipboard:
        text = subprocess.run(["pbpaste"], capture_output=True, text=True, check=True).stdout
    else:
        print("Paste the posting, then press Ctrl-D on a new line:")
        text = sys.stdin.read()
    if len(text.strip()) < 200:
        sys.exit("that looks too short to be a posting; nothing saved")
    path.write_text(text.strip() + "\n")
    print(f"saved {path} ({len(text):,} characters). Label it with: python -m eval.label")


def cmd_label(only: str | None) -> None:
    labels = read_labels()
    files = sorted(POSTINGS.glob("*.txt"))
    files = [p for p in files if p.stem != "README"]
    todo = [p for p in files if (p.stem == only if only else p.stem not in labels)]
    if only and not todo:
        sys.exit(f"no posting eval/postings/{only}.txt")
    if not todo:
        print("Every posting has a label. `python -m eval.label status` to review.")
        return
    print("Your profile: " + profile_summary())
    print(f"{len(todo)} to label. Labels save after each one, so you can stop anytime.")
    for p in todo:
        r = label_one(p.stem, p.read_text())
        if r == "quit":
            break
        if r:
            labels[p.stem] = r
            write_labels(labels)
            print(f"  saved: {r['expected_eligible']}, blockers {r['expected_blockers'] or 'none'}")
    cmd_status()


def cmd_status() -> None:
    labels = read_labels()
    have = {p.stem for p in POSTINGS.glob("*.txt") if p.stem != "README"}
    v = Counter(r["expected_eligible"] for r in labels.values())
    cats = Counter(c for r in labels.values() for c in r["expected_blockers"])
    print(f"\n{len(labels)} labeled, {len(have - labels.keys())} unlabeled, "
          f"{len(labels.keys() - have)} labels without a posting file")
    print(f"verdicts: yes {v['yes']} · no {v['no']} · check {v['check']}")
    print("blockers by type: " + (", ".join(f"{c} {n}" for c, n in cats.most_common()) or "none yet"))
    tips = []
    if len(labels) < 30:
        tips.append(f"add {30 - len(labels)} more postings")
    for k, want in (("no", 8), ("yes", 8), ("check", 5)):
        if v[k] < want:
            tips.append(f"more '{k}' postings (have {v[k]}, aim for {want}+)")
    if len(cats) < 3:
        tips.append("blockers of at least 3 different types (e.g. graduation, location, sponsorship)")
    print("to balance the set: " + ("; ".join(tips) if tips else "looks good"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", default="label", choices=["label", "add", "redo", "status"])
    ap.add_argument("id", nargs="?")
    ap.add_argument("--clipboard", action="store_true", help="add: read the posting from the Mac clipboard")
    a = ap.parse_args()
    if a.cmd == "add":
        if not a.id:
            sys.exit("usage: python -m eval.label add <id>")
        cmd_add(a.id, a.clipboard)
    elif a.cmd == "redo":
        if not a.id:
            sys.exit("usage: python -m eval.label redo <id>")
        cmd_label(a.id)
    elif a.cmd == "status":
        cmd_status()
    else:
        cmd_label(None)


if __name__ == "__main__":
    main()
