"""Resume support for long jobs on Kaggle (sessions end after ~12h, or can be cut at any time).

Design
  * Results are appended to ONE JSONL file, line by line, flushed + fsynced (a crash loses at most 1 line).
  * Every row has a unique "key" (e.g. "CVS-uBRP/H10S6/inst07/M2560/seed0"). On start, keys found in the
    current out_dir AND in any previous-run dirs (Kaggle input datasets) are skipped; old rows are copied into
    the new results.jsonl, so the newest run's output is always the complete, cumulative record.
  * A wall-clock budget makes the job stop gracefully *before* Kaggle kills it, and write state.json
    {"finished": false, ...}. The local driver (kaggle_cli.py --auto-resume) reads state.json, uploads the
    cumulative output as a private dataset and starts the next session, which picks it up automatically.
  * Training jobs use save_checkpoint()/latest_checkpoint() (atomic writes) the same way.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import time
from pathlib import Path


def find_resume_dirs(patterns=("/kaggle/input/**/state.json",)) -> list:
    """Directories (under Kaggle's read-only input mount) that hold a previous run's state.json."""
    out = []
    for pat in patterns:
        out += [str(Path(p).parent) for p in glob.glob(pat, recursive=True)]
    return sorted(set(out))


def _read_jsonl(path: Path) -> list:
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # ignore a torn last line left by a crash
    return rows


class ResumableRun:
    def __init__(self, out_dir, resume_dirs=(), budget_s: float | None = None, margin_s: float = 300.0):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.resume_dirs = [Path(d) for d in resume_dirs]
        self.t0 = time.time()
        self.deadline = None if budget_s is None else self.t0 + budget_s - margin_s
        self.results = self.out / "results.jsonl"
        rows = {}
        for d in self.resume_dirs + [self.out]:  # later sources win
            for r in _read_jsonl(d / "results.jsonl"):
                if "key" in r:
                    rows[r["key"]] = r
        self.done = set(rows)
        # rewrite the cumulative file atomically (copies old runs' rows into this run's output)
        tmp = self.results.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows.values()), encoding="utf-8")
        os.replace(tmp, self.results)
        self.n_inherited = len(rows)

    # ---- results ---------------------------------------------------------------------------------
    def is_done(self, key: str) -> bool:
        return key in self.done

    def record(self, row: dict) -> None:
        assert "key" in row, "every result row needs a unique 'key'"
        with open(self.results, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.done.add(row["key"])

    # ---- time budget -----------------------------------------------------------------------------
    def time_left(self) -> float:
        return float("inf") if self.deadline is None else self.deadline - time.time()

    def should_stop(self) -> bool:
        return self.time_left() <= 0

    # ---- checkpoints (atomic) --------------------------------------------------------------------
    def ckpt_path(self, name: str) -> Path:
        return self.out / "checkpoints" / name

    def save_checkpoint(self, name: str, writer) -> Path:
        """writer(path) must write the file (e.g. lambda p: torch.save(obj, p)). Written to a temp file then
        renamed, so a cut session never leaves a half-written checkpoint."""
        final = self.ckpt_path(name)
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp = final.with_name(final.name + ".tmp")
        writer(tmp)
        os.replace(tmp, final)
        return final

    def latest_checkpoint(self, name: str) -> Path | None:
        cands = [d / "checkpoints" / name for d in self.resume_dirs + [self.out]]
        cands = [c for c in cands if c.exists()]
        return max(cands, key=lambda c: c.stat().st_mtime) if cands else None

    def adopt_checkpoint(self, name: str) -> Path | None:
        """Copy the newest checkpoint into this run's output (so it is carried to the next session)."""
        src = self.latest_checkpoint(name)
        if src is None:
            return None
        dst = self.ckpt_path(name)
        if src.resolve() != dst.resolve():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        return dst

    # ---- driver ----------------------------------------------------------------------------------
    def run_items(self, items, fn, log=print) -> bool:
        """items: iterable of (key, payload). fn(payload) -> dict row (without key). Returns True if every
        item is done. Stops cleanly when the time budget is exhausted."""
        items = list(items)
        todo = [(k, p) for k, p in items if not self.is_done(k)]
        log(f"resume: {len(items) - len(todo)}/{len(items)} already done, {len(todo)} to run")
        for i, (k, p) in enumerate(todo, 1):
            if self.should_stop():
                log(f"time budget reached after {i - 1}/{len(todo)} new items; stopping for resume")
                break
            row = fn(p)
            row["key"] = k
            self.record(row)
            if i % max(1, len(todo) // 20) == 0 or i == len(todo):  # ~20 progress lines per session
                log(f"progress: {i}/{len(todo)} new items | last key {k} | {time.time() - self.t0:.0f}s elapsed")
        finished = all(self.is_done(k) for k, _ in items)
        self.finish(finished, n_total=len(items))
        return finished

    def finish(self, finished: bool, **extra) -> None:
        state = {"finished": bool(finished), "n_done": len(self.done), "n_inherited": self.n_inherited,
                 "wall_s": round(time.time() - self.t0, 1), **extra}
        (self.out / "state.json").write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
