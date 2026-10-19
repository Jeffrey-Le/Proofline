"""Deterministic eligibility checks. No LLM here on purpose.

The extractor turns prose into structured constraints; this module decides.
Keeping the decision in plain code makes blockers reproducible, unit-testable
and explainable ("Requires grad by 2026-06; you graduate 2026-12").
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from .schemas import Constraint, EligibilityFinding, Extraction


@dataclass
class Profile:
    graduation: str                     # YYYY-MM of degree conferral
    degree_level: str                   # BS | MS | PhD
    fields: list[str]                   # major + related fields
    willing_locations: list[str]
    open_to_relocation: bool
    needs_sponsorship: bool
    # Confirmed enrollment after graduation (e.g. admitted MS). None = no plan confirmed.
    next_program: Optional[dict] = None  # {"level": "MS", "start": "2027-09", "status": "applying|admitted|enrolled"}
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "Profile":
        return cls(**yaml.safe_load(Path(path).read_text()))

    def enrolled_on(self, ym: str) -> Optional[bool]:
        """Will I be a degree-seeking student in month `ym`? None = depends on an unconfirmed plan."""
        if ym <= self.graduation:
            return True
        np = self.next_program
        if np and np.get("start", "9999-12") <= ym:
            return True if np.get("status") in ("admitted", "enrolled") else None
        if np and np.get("status") == "applying":
            return None
        return False


# ---------- helpers ----------
_WS = re.compile(r"\s+")


_TRANS = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                        "\u2013": "-", "\u2014": "-", "\u00a0": " "})
_WRAP = " \t\n\"'`<>[]()*.,;:-"
_ELLIPSIS = re.compile(r"\.\.\.+|\u2026")
_LINE_ID = re.compile(r"^(\[[a-z0-9-]{1,12}\]\s*|[a-z]{1,8}-?\d{1,3}\s*:\s+)", re.I)  # "[s-01] " or "b03: "


def norm(s: str) -> str:
    return _WS.sub(" ", s.translate(_TRANS)).strip().lower()


_MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


def human(ym: Optional[str]) -> str:
    """2026-12 -> Dec 2026."""
    try:
        y, m = ym.split("-")
        return f"{_MONTHS[int(m) - 1]} {y}"
    except Exception:
        return ym or "an unknown date"


def grounded(quote: str, source: str) -> bool:
    """Is `quote` literally present in `source`?

    Tolerates what models add around a real quote (quotation marks, brackets, a leading line id,
    trailing punctuation, case and whitespace) and "..." between fragments, as long as every
    fragment appears in the source, in order. The words themselves must match exactly.
    """
    q = _LINE_ID.sub("", norm(quote).strip(" \t\n\"'`*")).strip(_WRAP)
    src = norm(source)
    parts = [p.strip(_WRAP) for p in _ELLIPSIS.split(q)]
    parts = [p for p in parts if p]
    if not parts or sum(len(p) for p in parts) < 4:
        return False
    pos = 0
    for p in parts:
        i = src.find(p, pos)
        if i < 0:
            return False
        pos = i + len(p)
    return True


def _mk(c: Constraint, status: str, reason: str, posting: str) -> EligibilityFinding:
    ok = grounded(c.source_quote, posting)
    if not ok and status == "blocker":
        # Never hard-block on a requirement we can't find in the posting text.
        status, reason = "warning", f"{reason} We couldn't find this exact wording in the posting, so check it yourself."
    return EligibilityFinding(category=c.category, kind=c.kind, status=status, reason=reason,
                              source_quote=c.source_quote, quote_grounded=ok)


def _fail(c: Constraint) -> str:
    return "blocker" if c.kind == "minimum" else "warning"


# ---------- one rule per category ----------
def _graduation(c, p: Profile, ex: Extraction):
    g = p.graduation
    if c.grad_after and g < c.grad_after:
        return _fail(c), f"Requires graduating in {human(c.grad_after)} or later; you graduate in {human(g)}."
    if c.grad_before and g > c.grad_before:
        return _fail(c), f"Requires graduating by {human(c.grad_before)}; you graduate in {human(g)}."
    if not (c.grad_after or c.grad_before):
        return "unknown", "The posting mentions a graduation requirement but no clear dates."
    return "pass", f"You graduate in {human(g)}, within the required window."


def _enrollment_at(ym: Optional[str], p: Profile, c, what: str):
    if not ym:
        return "unknown", f"Requires {what}, but the posting doesn't give clear program dates."
    e = p.enrolled_on(ym)
    if e is True:
        return "pass", f"You'll still be enrolled in {human(ym)}."
    if e is None:
        np = p.next_program or {}
        return "warning", (f"Requires {what} ({human(ym)}). That depends on your {np.get('level', 'next')} "
                           f"program starting {human(np.get('start'))}, which you're still {np.get('status', 'applying')} to."
                           if np.get("status") == "applying" else
                           f"Requires {what} ({human(ym)}). That depends on your next program.")
    return _fail(c), f"Requires {what} ({human(ym)}), but you graduate in {human(p.graduation)} with no next program planned."


def _enrollment(c, p, ex):
    return _enrollment_at(ex.program_start, p, c, "being enrolled during the program")


def _return(c, p, ex):
    end = ex.program_end or ex.program_start
    if end:  # must be enrolled the term after the internship ends
        y, m = map(int, end.split("-"))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        end = f"{y:04d}-{m:02d}"
    return _enrollment_at(end, p, c, "returning to school after the program")


_LEVEL = {"BS": 1, "BA": 1, "MS": 2, "MENG": 2, "PHD": 3}


def _degree_level(c, p, ex):
    acc = {l.upper().replace(".", "") for l in c.degree_levels}
    if not acc:
        return "unknown", "The posting mentions a degree requirement but not which degrees."
    mine = {p.degree_level.upper()}
    if p.next_program and p.next_program.get("status") in ("admitted", "enrolled"):
        mine.add(p.next_program["level"].upper())
    if mine & acc:
        return "pass", f"Accepts {', '.join(sorted(acc))}; you have or are pursuing {', '.join(sorted(mine))}."
    return _fail(c), f"Accepts {', '.join(sorted(acc))}; you have {', '.join(sorted(mine))}."


_OPEN_FIELD = re.compile(r"\b(related|relevant|equivalent|similar|any|no specific)\b")


def _degree_field(c, p, ex):
    s = norm(c.summary + " " + c.source_quote)
    # Open-ended wording ("or a related field", "a field relevant to the role", "or equivalent
    # experience", "any technical field") doesn't restrict the field, so it passes.
    if _OPEN_FIELD.search(s) or any(norm(f) in s for f in p.fields):
        return "pass", "Your field of study fits the requirement."
    return "warning", f"Check whether {p.fields[0]} counts for this field requirement."


def _location(c, p, ex):
    if c.remote_ok:
        return "pass", "Remote work is allowed."
    locs = [norm(l) for l in c.locations]
    if not locs:
        return "unknown", "The posting mentions a location requirement but no specific places."
    hit = [l for l in locs if any(norm(w) in l or l in norm(w) for w in p.willing_locations)]
    if hit:
        return "pass", f"In {', '.join(c.locations)}, which is on your list."
    if p.open_to_relocation:
        return "pass", f"On-site in {', '.join(c.locations)}; you're open to relocation."
    return _fail(c), f"On-site in {', '.join(c.locations)}, which isn't on your list."


def _auth(c, p, ex):
    if p.needs_sponsorship and c.sponsorship_available is False:
        return _fail(c), "This role doesn't offer visa sponsorship, and you need it."
    return "pass", "No conflict with your work authorization."


def _other(c, p, ex):
    return "unknown", f"Check this requirement yourself: {c.summary}"


RULES = {
    "graduation_window": _graduation, "enrollment": _enrollment, "return_to_school": _return,
    "degree_level": _degree_level, "degree_field": _degree_field, "location": _location,
    "work_authorization": _auth, "program_dates": lambda c, p, ex: ("pass", "Informational."),
    "other": _other,
}


def check(ex: Extraction, profile: Profile, posting: str) -> list[EligibilityFinding]:
    out = []
    for c in ex.constraints:
        status, reason = RULES.get(c.category, _other)(c, profile, ex)
        out.append(_mk(c, status, reason, posting))
    return out
