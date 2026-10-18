"""Structured outputs for each agent, plus the final report.

Optional fields default to None because ADK stores agent output in session state
with None values dropped; callbacks always re-validate through these models.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Kind = Literal["minimum", "preferred"]
Category = Literal[
    "graduation_window",
    "enrollment",
    "return_to_school",
    "degree_level",
    "degree_field",
    "location",
    "work_authorization",
    "program_dates",
    "other",
]


# ---------- Agent 1: extractor ----------
class Constraint(BaseModel):
    """A hard eligibility condition (who may apply), not a skill."""

    category: Category
    kind: Kind = Field(description="'minimum' if listed under minimum/required qualifications, else 'preferred'")
    summary: str = Field(description="One-line normalized restatement")
    grad_after: Optional[str] = Field(default=None, description="Earliest allowed graduation, YYYY-MM, or null")
    grad_before: Optional[str] = Field(default=None, description="Latest allowed graduation, YYYY-MM, or null")
    degree_levels: list[str] = Field(description="Accepted levels from: BS, MS, PhD. Empty if not stated")
    locations: list[str] = Field(description="Office cities/countries. Empty if not stated")
    remote_ok: Optional[bool] = None
    sponsorship_available: Optional[bool] = None
    source_quote: str = Field(description="Exact span copied from the posting that states this")


class Qualification(BaseModel):
    """A skill/experience requirement to match against resume evidence."""

    id: str = Field(description="q1, q2, ...")
    kind: Kind
    text: str = Field(description="The qualification as one checkable claim")
    source_quote: str = Field(description="Exact span copied from the posting")


class Extraction(BaseModel):
    company: Optional[str] = None
    title: Optional[str] = None
    program_start: Optional[str] = Field(default=None, description="Internship/job start, YYYY-MM, or null (e.g. Summer 2027 -> 2027-06)")
    program_end: Optional[str] = Field(default=None, description="End, YYYY-MM, or null")
    constraints: list[Constraint]
    qualifications: list[Qualification]


# ---------- Agent 2: matcher ----------
class Match(BaseModel):
    qual_id: str
    verdict: Literal["supported", "partial", "none"]
    bullet_id: Optional[str] = Field(default=None, description="ID of the single best evidence bullet, or null")
    evidence_quote: Optional[str] = Field(default=None, description="Exact substring of that bullet, or null")
    rationale: str


class MatchSet(BaseModel):
    matches: list[Match]


# ---------- Resume splitter (visitor sessions) ----------
class ResumeBullet(BaseModel):
    text: str = Field(description="Copied verbatim from the resume")
    source: str = Field(description="Role and organization or project this bullet belongs to")


class BulletSet(BaseModel):
    bullets: list[ResumeBullet]


# ---------- Visitor profile (validated API input) ----------
class NextProgram(BaseModel):
    level: Literal["BS", "MS", "PhD"]
    start: str = Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")
    status: Literal["applying", "admitted", "enrolled"]


class ProfileIn(BaseModel):
    graduation: str = Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")
    degree_level: Literal["BS", "MS", "PhD"]
    fields: list[str] = Field(max_length=6)
    willing_locations: list[str] = Field(max_length=12)
    open_to_relocation: bool
    needs_sponsorship: bool
    next_program: Optional[NextProgram] = None


# ---------- Agent 3: verifier ----------
class Verdict(BaseModel):
    qual_id: str
    accepted: bool
    supported_level: Literal["supported", "partial", "none"]
    reason: str


class VerdictSet(BaseModel):
    verdicts: list[Verdict]


# ---------- Deterministic outputs / final report ----------
Status = Literal["pass", "blocker", "warning", "unknown"]


class EligibilityFinding(BaseModel):
    category: str
    kind: str
    status: Status
    reason: str
    source_quote: str
    quote_grounded: bool


class FitResult(BaseModel):
    qual_id: str
    kind: str
    qualification: str
    source_quote: str
    final: Literal["supported", "partial", "none"]
    bullet_id: Optional[str]
    evidence: Optional[str]
    rejected_by: Optional[Literal["grounding_check", "verifier"]]
    note: str


class Report(BaseModel):
    company: Optional[str]
    title: Optional[str]
    eligible: Literal["yes", "no", "check"]
    blockers: list[EligibilityFinding]
    warnings: list[EligibilityFinding]
    eligibility: list[EligibilityFinding]
    fit: list[FitResult]
    fit_score: float  # share of minimum quals with verified support
    notes: list[str] = Field(default_factory=list)  # e.g. a step had to be skipped
