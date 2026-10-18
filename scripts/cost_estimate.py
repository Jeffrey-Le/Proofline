"""Estimate a month of running Proofline publicly.

    python -m scripts.cost_estimate                  # planning estimate (assumed tokens)
    python -m scripts.cost_estimate --measured       # uses eval/runs/cache.jsonl averages

Token counts below are planning assumptions. After your first eval run, use --measured:
it replaces the per-check numbers with what your traces actually recorded.
"""
import argparse
import json
from pathlib import Path

from proofline import config

P = config.PRICES
LITE, FLASH, EMB = "gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-embedding-001"

# (model, input tokens, output tokens incl. thinking) per call
CHECK = [
    ("extractor", LITE, 2_200, 1_800),   # prompt + ~1.5k-token posting -> structured JSON
    ("matcher",   LITE, 3_000,   900),   # ReAct turn 1: seed evidence, decides to search
    ("matcher",   LITE, 3_600, 1_200),   # ReAct turn 2: with search results, writes notes
    ("formatter", LITE, 1_000,   600),   # notes -> JSON
    ("verifier",  FLASH, 1_600, 1_600),  # surviving claims only, MEDIUM thinking
    ("embed",     EMB,     150,     0),  # qualification queries
]
SESSION = [
    ("splitter",  LITE, 2_200, 2_200),   # resume in, verbatim bullets out
    ("embed",     EMB,   1_000,     0),
]


def cost(calls):
    return sum(i * P[m][0] / 1e6 + o * P[m][1] / 1e6 for _, m, i, o in calls)


def measured_check_cost():
    f = Path("eval/runs/cache.jsonl")
    costs = [r["trace"]["cost_usd"] for r in map(json.loads, f.open()) if r.get("trace", {}).get("cost_usd")]
    return (sum(costs) / len(costs), len(costs)) if costs else (None, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measured", action="store_true")
    ap.add_argument("--days", type=int, default=30)
    a = ap.parse_args()

    per_check, per_session = cost(CHECK), cost(SESSION)
    if a.measured:
        m, n = measured_check_cost()
        if m:
            print(f"measured: ${m:.4f}/check over {n} eval runs (assumed was ${per_check:.4f})")
            per_check = m
    print(f"per check:   ${per_check:.4f}")
    print(f"per resume:  ${per_session:.4f}\n")

    # visitors/day, checks per visitor, share of visitors who save their own resume
    scenarios = {
        "quiet (recruiters + a few friends)": (5, 2, 0.4),
        "shared in a Slack / Discord":         (30, 3, 0.6),
        "front page of a subreddit":           (300, 3, 0.6),
    }
    cap = config.DAILY_BUDGET_USD
    print(f"{'scenario':38} {'checks/day':>10} {'$/day':>8} {'$/month':>9}   with ${cap:.2f}/day cap")
    for name, (visitors, cpv, share) in scenarios.items():
        checks, sessions = visitors * cpv, visitors * share
        day = checks * per_check + sessions * per_session
        print(f"{name:38} {checks:>10.0f} {day:>8.2f} {day * a.days:>9.2f}   "
              f"${min(day, cap) * a.days:.2f}/month{'  (cap hit, demo pauses daily)' if day > cap else ''}")
    print(f"\nHard ceiling for public traffic: ${cap:.2f} x {a.days} days = ${cap * a.days:.2f} "
          "(plus whatever you spend with your owner link).")
    print("Cloud Run, Firestore, Secret Manager: inside free tiers at these volumes (see README).")


if __name__ == "__main__":
    main()
