import json
import time

from brp.resume import ResumableRun, find_resume_dirs


def items(n):
    return [(f"k{i:02d}", i) for i in range(n)]


def test_interrupted_run_resumes_without_duplicates(tmp_path):
    out1, out2 = tmp_path / "run1", tmp_path / "run2"
    calls = []

    def fn(i):
        calls.append(i)
        time.sleep(0.02)
        return {"val": i * i}

    r1 = ResumableRun(out1, budget_s=0.25, margin_s=0)  # stops partway
    assert r1.run_items(items(30), fn, log=lambda *_: None) is False
    n1 = len(calls)
    assert 0 < n1 < 30
    assert json.loads((out1 / "state.json").read_text())["finished"] is False

    r2 = ResumableRun(out2, resume_dirs=[out1])  # next session: no budget -> finishes
    assert r2.n_inherited == n1
    assert r2.run_items(items(30), fn, log=lambda *_: None) is True
    assert len(calls) == 30  # nothing recomputed
    rows = [json.loads(x) for x in (out2 / "results.jsonl").read_text().splitlines()]
    assert sorted(r["key"] for r in rows) == [k for k, _ in items(30)]  # cumulative, no duplicates
    assert all(r["val"] == int(r["key"][1:]) ** 2 for r in rows)
    assert json.loads((out2 / "state.json").read_text())["finished"] is True


def test_torn_last_line_is_ignored(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "results.jsonl").write_text('{"key": "a", "v": 1}\n{"key": "b", "v"', encoding="utf-8")
    r = ResumableRun(out)
    assert r.is_done("a") and not r.is_done("b")
    r.record({"key": "b", "v": 2})
    lines = (out / "results.jsonl").read_text().splitlines()
    assert [json.loads(x)["key"] for x in lines] == ["a", "b"]


def test_checkpoint_atomic_and_latest(tmp_path):
    o1, o2 = tmp_path / "run1", tmp_path / "run2"
    r1 = ResumableRun(o1)
    r1.save_checkpoint("model.bin", lambda p: p.write_bytes(b"v1"))
    assert not list((o1 / "checkpoints").glob("*.tmp"))
    time.sleep(0.05)
    r2 = ResumableRun(o2, resume_dirs=[o1])
    assert r2.latest_checkpoint("model.bin").read_bytes() == b"v1"
    r2.adopt_checkpoint("model.bin")
    r2.save_checkpoint("model.bin", lambda p: p.write_bytes(b"v2"))
    assert r2.latest_checkpoint("model.bin").read_bytes() == b"v2"


def test_find_resume_dirs(tmp_path):
    d = tmp_path / "a" / "b"
    d.mkdir(parents=True)
    (d / "state.json").write_text("{}")
    assert find_resume_dirs((str(tmp_path / "**" / "state.json"),)) == [str(d)]
