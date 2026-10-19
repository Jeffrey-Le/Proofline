# Proofline

**Live:** <your Cloud Run URL> · try it with the built-in sample resume, or save your own for 24 hours.

Paste a job posting. Proofline extracts the hard eligibility gates, checks them against a
profile with deterministic rules, matches each qualification to resume evidence with RAG,
and has a verifier agent reject any match the source text doesn't support. Visitors can use
a fictional sample resume or save their own for 24 hours. The owner's real resume is only
reachable through a private link.

```
posting ─▶ [extractor LLM] ─▶ eligibility rules (Python) ─▶ blockers / warnings
                 │
                 └─ qualifications ─▶ embed + vector search ─▶ [matcher LLM] ⇄ search_resume tool (ReAct, ≤6 calls)
                                                                    │ notes ─▶ [formatter LLM] ─▶ JSON
                     grounding gate (Python: quote ⊂ cited bullet?) ◀┘
                                    │ survivors only
                               [verifier LLM] ── may downgrade, never upgrade ─▶ report
every step ─▶ span {latency, tokens, cost} ─▶ one JSON log line per request (Cloud Logging / Trace)

visitor resume ─▶ [splitter LLM] ─▶ keep only lines found word for word ─▶ embed ─▶ Firestore
                  (session_id + expire_at; TTL deletes after 24 h; searches filtered by session_id)
```

**Design rule:** LLMs read and judge language; code makes every decision that should be
reproducible. A blocker is always a rule firing on a quote that literally appears in the posting,
so no hallucinated requirement can ever hard-block you.

## Run locally

```bash
python -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt
export GEMINI_API_KEY=...            # paid-tier key once visitors can paste resumes (see below)
python -m pytest -q                  # 42 tests, no API calls (fake LLM)
cp data/profile.example.yaml data/profile.yaml && cp data/bullets.example.yaml data/bullets.yaml  # then edit; both are gitignored
python -m scripts.load_bullets       # your resume (private) into local LanceDB
python -m scripts.load_bullets --sample  # the fictional public sample
python -m scripts.check eval/postings/some-posting.txt
uvicorn app:app --reload             # http://localhost:8000
```

Model IDs, thinking levels and prices live in `proofline/config.py` (env-overridable).
`python -m scripts.cost_estimate` projects a month of spend; add `--measured` after an eval
run to use your real per-check cost instead of the planning assumptions.

## Deploy

Step by step, with a checklist and troubleshooting: see `DEPLOY.md`.

`PROJECT=... BILLING=... GEMINI_API_KEY=... ./deploy.sh` creates a $20 budget alert,
Firestore with a session-filtered vector index and TTL policies, secrets, loads your bullets,
and deploys to Cloud Run (scale-to-zero, max 1 instance). Open `URL/#token=...` once to store
your owner token, which skips the public limits.

## Running it for the public

| Concern | What the code does |
|---|---|
| Gemini free-tier terms forbid other people's personal data | Use a paid-tier key. |
| Visitors' resumes | Splitter keeps only lines found word for word; the full text, contact details and posting text are never stored or logged. Rows carry `expire_at`; Firestore TTL deletes them after 24 h, reads also check expiry, and "Delete it now" removes them at once. |
| One visitor reading another's resume | Every vector search is pre-filtered by a random 128-bit `session_id`; ids are format-checked. |
| Prompt injection in postings or resumes | Untrusted text is fenced and the prompts say never to follow it; the grounding gate means injected claims still need a verbatim quote to count. |
| Runaway cost | `DAILY_BUDGET_USD` (default $0.50) of *measured* spend pauses public traffic until midnight PT; per-IP hourly limit; max-instances=1. Ceiling ≈ $15/month plus your own use. |
| Wrong "not eligible" | Every verdict shows its quote, and the page says it's a second opinion. |

Measured on the 30-posting eval: $0.016 per check (planning estimate for a saved resume: $0.006), so
a quiet month (≈10 checks/day) is about $5 and the hard ceiling is $15. Cloud Run, Firestore
and Secret Manager stay inside their free tiers at these volumes. The biggest lever is output
tokens: moving the verifier to Flash-Lite with LOW thinking cuts a check to roughly $0.011.

## RL environment (`rl/`)

The matching task doubles as an RL environment with a verifiable reward: synthetic resumes
and postings with exact ground truth, rewards computed by code, prompt injections included,
and held-out skill families for transfer. Includes a GRPO + LoRA training script (TRL) and an
evaluator that compares a base model, the trained model and the production Gemini model.
No visitor data is ever used. See `rl/README.md`.

## Evaluate

Labels are written blind, before running Proofline, so the eval can't grade the model against itself.

```bash
python -m eval.label add google-fde-2027 --clipboard   # save a posting (or paste, then Ctrl-D)
python -m eval.label                                   # label every unlabeled posting
python -m eval.label status                            # check the set is balanced
python -m eval.run_eval                                # run and score
```

Reports blocker recall/precision, postings with a missed blocker (the expensive failure), false
blocks, soft-flag recall, verdict accuracy, p50/p95 latency, tokens and cost per request, and how
many fit claims the grounding gate vs. the verifier rejected. Writes `eval/runs/results.md` with the
table below and every miss. Results are cached, so re-scoring is free; `--fresh` after prompt changes.

## Results

30 real Summer 2027 internship and early-career postings (Google, Meta, Anthropic, OpenAI, Amazon,
Jane Street, Waymo, robotics startups and others), checked against the owner's profile. Labels
were drafted by a different model (Claude) than the one being tested, from the posting text
alone and before any Proofline output was seen.

| Metric | First run | After fix |
|---|---|---|
| Overall verdict accuracy (yes / no / check) | 25/30 (83%) | **29/30 (97%)** |
| False blocks (eligible postings marked "not eligible") | 0/30 | **0/30** |
| Soft-flag recall (warnings worth surfacing) | 84.6% | 84.6% |
| Proposed resume matches rejected: grounding gate / verifier | 15 / 82 of 167 | 10 / 85 of 165 (58%) |
| Latency p50 / p95 | 20.2 s / 32.8 s | 28.8 s / 80.5 s |
| Measured cost per check | $0.0158 | **$0.0156–0.0158** |

**What the first run found.** Four postings (three Anthropic, one Parallel) say "a field relevant to
the role". The field-of-study rule only treated "related" as open-ended, so it warned on a
requirement that isn't one, turning four "yes" verdicts into "check". The rule now accepts
open-ended wording (related, relevant, equivalent, any) and has a regression test. Prompts were not
changed between runs, so the second run isn't tuned to this set.

**Remaining miss.** Jane Street's ML internship asks for "an undergraduate or PhD student". The
extractor treated this as a description rather than an eligibility gate, so it returned "yes"
where a December graduate heading into an MS should see "check".

**Limits of this eval, stated plainly.**
- *Blocker recall isn't measured yet.* None of these 30 postings hard-blocks this profile (open to
  relocation, no sponsorship needed, every role accepts a BS), so the most important number has no
  test cases. A set of postings with real blockers (PhD-only, other graduation windows, clearance) is next.
- *Years of experience isn't an eligibility gate.* 9 of 30 postings ask for 2–3+ years. Proofline
  scores this as fit, not eligibility, so those count as "yes" here.
- *Rejected matches aren't graded.* 58% of proposed resume matches were rejected by the grounding
  gate or the verifier. That shows the checks are active, not that every rejection was right;
  grading them needs fit labels.
- *Tail latency varies.* p95 rose from 33 s to 80 s between runs with no code change on the hot
  path, consistent with model-side latency; the service timeout is 120 s.

## Roadmap

- Migrate `SequentialAgent` to the ADK `Workflow` graph, with the rule steps as function nodes.
- A verifier-only eval of hand-labeled (qualification, bullet, should-support) pairs to report verifier precision directly.
- Train a small open model on the RL environment and compare it to the production model on the real labeled set.
