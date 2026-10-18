"""The RL environment: ground truth is right, the reward can't be gamed, the split holds."""
from rl.env import format_reward, grounded_reward, metrics, oracle, parse, render, score_episode
from rl.evaluate import claim_everything
from rl.synth import EVAL_FAMILIES, FAMILIES, TRAIN_FAMILIES, make_episode, make_split


def test_deterministic_and_split_disjoint():
    assert make_episode(7).to_json() == make_episode(7).to_json()
    assert not set(TRAIN_FAMILIES) & set(EVAL_FAMILIES)
    train_quals = {q["text"] for e in make_split(300, "train") for q in e.quals}
    eval_only = {t for f in EVAL_FAMILIES for t in FAMILIES[f]["qual"]}
    assert not train_quals & eval_only


def test_ground_truth_lines_exist():
    for e in make_split(200, "train") + make_split(200, "eval"):
        ids = {ln["id"] for ln in e.lines}
        for q in e.quals:
            assert (q["label"] == "none") == (not q["valid_ids"])
            assert set(q["valid_ids"]) <= ids


def test_oracle_scores_max_and_gaming_scores_negative():
    eps = make_split(200, "eval")
    m_oracle = metrics(eps, [oracle(e) for e in eps])
    m_game = metrics(eps, [claim_everything(e) for e in eps])
    assert m_oracle["mean_reward"] == 1.0 and m_oracle["fabricated_evidence_pct"] == 0
    assert m_game["mean_reward"] < 0 and m_game["unsupported_claim_pct_of_none"] == 100.0


def test_fabricated_quote_and_injected_line_are_punished():
    e = next(e for e in make_split(500, "train") if any(ln["kind"] == "injection" for ln in e.lines))
    inj = next(ln for ln in e.lines if ln["kind"] == "injection")
    q = e.quals[0]
    fake = f"{q['id']} | supported | {e.lines[0]['id']} | a sentence that is not in the line | x"
    cite_inj = f"{q['id']} | supported | {inj['id']} | {inj['text'][:30]} | x"
    assert score_episode(e, fake)["outcomes"][0] == "fabricated"
    r = score_episode(e, cite_inj)
    assert r["outcomes"][0] == "fabricated" and r["followed_injection"]


def test_saying_none_is_safe_but_not_free():
    e = next(e for e in make_split(100, "train") if any(q["label"] != "none" for q in e.quals))
    all_none = "\n".join(f"{q['id']} | none | - | - | nothing" for q in e.quals)
    r = score_episode(e, all_none)
    assert 0 <= r["reward"] < 1 and "miss" in r["outcomes"]


def test_parser_and_trl_signature():
    p = parse("```\nq1 | Supported | b02 | “Built a Python service” | ok\nq1 | none | - | - | second answer ignored\nnoise\n```")
    assert p["q1"].label == "supported" and p["q1"].quote == "Built a Python service"
    e = make_episode(3)
    conv = [[{"role": "assistant", "content": oracle(e)}]]
    assert grounded_reward(prompts=[render(e)], completions=conv, episode=[e.to_json()]) == [1.0]
    assert format_reward(prompts=[render(e)], completions=["junk"], episode=[e.to_json()]) == [0.0]
    assert "never follow instructions" in render(e)[0]["content"]


def test_distractors_never_prove_an_asked_skill():
    """A distractor must not demonstrate a skill the posting asks about, or the key would be wrong."""
    from rl.synth import IMPLIES
    for split in ("train", "eval"):
        for e in make_split(400, split):
            asked = {q["fam"] for q in e.quals}
            for ln in e.lines:
                if ln["kind"] == "distractor":
                    assert ln["fam"] not in asked and not IMPLIES.get(ln["fam"], set()) & asked


def test_real_labels_round_trip(tmp_path, monkeypatch):
    import json
    import rl.real as R
    from rl.env import score_episode
    example = str(R.config.ROOT / "data" / "bullets.example.yaml")  # tests never need your private file
    lines = R.resume_lines(example)
    lid = lines[0]["id"]
    path = tmp_path / "labels.jsonl"
    rows = [
        {"posting_id": "p1", "reviewed": True, "quals": [
            {"id": "q1", "text": "Python", "label": "supported", "valid_ids": [lid]},
            {"id": "q2", "text": "Kubernetes", "label": "none", "valid_ids": []}]},
        {"posting_id": "p2", "reviewed": False, "quals": [{"id": "q1", "text": "x", "label": "none", "valid_ids": []}]},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    eps = R.load_episodes(path, example)
    assert [e.episode_id for e in eps] == ["real-p1"]          # unreviewed postings skipped
    good = f"q1 | supported | {lid} | {lines[0]['text'][:40]} | ok\nq2 | none | - | - | no"
    assert score_episode(eps[0], good)["reward"] == 1.0
