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
python -m pytest -q                  # 37 tests, no API calls (fake LLM)
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

Planning estimate at current prices (with the ReAct loop): about $0.02 per check and $0.006 per saved resume, so
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

Copy `eval/labels.example.jsonl` to `eval/labels.jsonl`, add one line per posting, put posting
text in `eval/postings/`, then `python -m eval.run_eval`. Reports blocker recall/precision,
postings with a missed blocker (the expensive failure), false blocks, soft-flag recall,
verdict accuracy, p50/p95 latency, tokens and cost per request, and how many fit claims the
grounding gate vs. the verifier rejected. Results are cached so re-scoring is free.

## Results

Evaluation on 30 hand-labeled postings is in progress; numbers will be added here once measured.

| Metric | Value |
|---|---|
| Blocker recall (postings with a real blocker that was caught) | pending |
| False blocks (eligible postings marked not eligible) | pending |
| Fit claims rejected: grounding gate / verifier | pending |
| Latency p50 / p95 | pending |
| Cost per check | pending |

## Roadmap

- Migrate `SequentialAgent` to the ADK `Workflow` graph, with the rule steps as function nodes.
- A verifier-only eval of hand-labeled (qualification, bullet, should-support) pairs to report verifier precision directly.
- Train a small open model on the RL environment and compare it to the production model on the real labeled set.
