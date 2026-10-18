"""Embed resume bullets and load them into the vector store.

    python -m scripts.load_bullets                  # your real resume (data/bullets.yaml)
    python -m scripts.load_bullets --sample         # the fictional public sample
    VECTOR_BACKEND=firestore python -m scripts.load_bullets [--sample]

Built-in resumes never expire. Only bullets marked verified: true are loaded.
"""
import argparse
import sys

import yaml

from proofline import config
from proofline.store import embed, get_store


def load(path: str) -> list[dict]:
    rows = yaml.safe_load(open(path))
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        sys.exit(f"duplicate bullet ids in {path}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="load the fictional sample resume instead of yours")
    a = ap.parse_args()

    if a.sample:
        sid, rows = config.SAMPLE_SESSION, load(config.SAMPLE_BULLETS_PATH)
        profile = yaml.safe_load(open(config.SAMPLE_PROFILE_PATH))
    else:
        sid, rows = config.OWNER_SESSION, load(config.BULLETS_PATH)
        profile = yaml.safe_load(open(config.PROFILE_PATH))  # stored with the bullets, never in the app build

    keep = [r for r in rows if r.get("verified") and not r["text"].startswith("<")]
    if not keep:
        sys.exit("no verified bullets to load")
    vecs = embed([r["text"] for r in keep], "RETRIEVAL_DOCUMENT")
    data = [{"id": r["id"], "text": r["text"], "source": r["source"], "vector": v} for r, v in zip(keep, vecs)]
    get_store().put_session(sid, data, profile=profile, expire_at=None)
    print(f"loaded {len(keep)} bullets as '{sid}' into {config.VECTOR_BACKEND} "
          f"(skipped {len(rows) - len(keep)} unverified or placeholder)")


if __name__ == "__main__":
    main()
