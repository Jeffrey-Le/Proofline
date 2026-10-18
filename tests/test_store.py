from proofline.store import LanceStore, _l2


def test_lancedb_roundtrip(tmp_path):
    s = LanceStore(str(tmp_path / "db"))
    rows = [{"id": f"b{i}", "text": t, "source": "x", "vector": _l2(v)}
            for i, (t, v) in enumerate([("rag", [1, 0, 0]), ("go", [0, 1, 0]), ("rag-ish", [.9, .1, 0])])]
    s.put_session("owner", rows, None, None)
    hits = s.search(_l2([1, 0, 0]), 2, "owner")
    assert [h.id for h in hits] == ["b0", "b2"] and abs(hits[0].score - 1) < 1e-5
