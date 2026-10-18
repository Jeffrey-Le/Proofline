"""Evaluate a policy on held-out GroundedMatch episodes (unseen skill families).

    python -m rl.evaluate --policy oracle                       # reward ceiling (should be 1.0)
    python -m rl.evaluate --policy claim_everything             # the gaming baseline
    python -m rl.evaluate --policy gemini:gemini-3.5-flash-lite # what production uses today
    python -m rl.evaluate --policy hf:Qwen/Qwen2.5-1.5B-Instruct            # base model
    python -m rl.evaluate --policy hf:runs/grounded-match --base Qwen/Qwen2.5-1.5B-Instruct  # after GRPO

Writes rl/results/<policy>.json so the before/after table in rl/README.md is reproducible.
"""
import argparse
import json
import re
from pathlib import Path

from .env import SYSTEM, metrics, oracle, render
from .synth import Episode, make_split


def claim_everything(ep: Episode) -> str:
    """Says every qualification is supported, citing the first line. What reward hacking looks like."""
    first = ep.lines[0]
    return "\n".join(f"{q['id']} | supported | {first['id']} | {first['text'][:40]} | looks good" for q in ep.quals)


def gemini_policy(model: str):
    from google import genai
    from google.genai import types
    client = genai.Client()

    def run(ep: Episode) -> str:
        msgs = render(ep)
        r = client.models.generate_content(
            model=model, contents=msgs[1]["content"],
            config=types.GenerateContentConfig(system_instruction=SYSTEM, temperature=0.0))
        return r.text or ""
    return run


def hf_policy(path: str, base: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base or path)
    model = AutoModelForCausalLM.from_pretrained(base or path, torch_dtype=torch.bfloat16, device_map="auto")
    if base:  # path is a LoRA adapter on top of base
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, path)
    model.eval()

    def run(ep: Episode) -> str:
        ids = tok.apply_chat_template(render(ep), add_generation_prompt=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=320, do_sample=False)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--base", default=None, help="base model when --policy hf:<lora-adapter-dir>")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--real", action="store_true", help="score on your reviewed real postings (rl/real.py)")
    a = ap.parse_args()

    if a.real:
        from .real import load_episodes
        eps = load_episodes()
    else:
        eps = make_split(a.n, "eval")
    if a.policy == "oracle":
        fn = oracle
    elif a.policy == "claim_everything":
        fn = claim_everything
    elif a.policy.startswith("gemini:"):
        fn = gemini_policy(a.policy.split(":", 1)[1])
    elif a.policy.startswith("hf:"):
        fn = hf_policy(a.policy.split(":", 1)[1], a.base)
    else:
        raise SystemExit("unknown policy")

    completions = [fn(e) for e in eps]
    m = {"policy": a.policy, **metrics(eps, completions)}
    name = re.sub(r"[^a-zA-Z0-9.-]+", "_", a.policy) + ("_real" if a.real else "")
    m["dataset"] = "real" if a.real else "synthetic-heldout"
    out = Path(__file__).parent / "results" / (name + ".json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(m, indent=2))
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
