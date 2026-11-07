"""Job-side logging shared by all Kaggle jobs (and usable locally).

Every line goes to stdout (flushed, so `kernels logs -f` shows it live) AND to <out_dir>/run.log, with a tag:
[PROGRESS] [RESULT] [SUMMARY] [WARN] [ERROR]. Also writes env.json and summary.json (see CLAUDE.md, Mục 3).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import platform
import subprocess
from pathlib import Path


class JobLog:
    def __init__(self, out_dir="/kaggle/working"):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self._f = open(self.out / "run.log", "a", encoding="utf-8")
        self.env_info = {"started": dt.datetime.now().isoformat(), "python": platform.python_version()}

    def log(self, tag: str, msg: str) -> None:
        line = f"[{dt.datetime.now().strftime('%H:%M:%S')}] [{tag}] {msg}"
        print(line, flush=True)
        self._f.write(line + "\n")
        self._f.flush()

    def progress(self, msg): self.log("PROGRESS", msg)
    def result(self, obj): self.log("RESULT", json.dumps(obj, ensure_ascii=False))
    def warn(self, msg): self.log("WARN", msg)
    def error(self, msg): self.log("ERROR", msg)

    def git_head(self, repo) -> str:
        sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        self.env_info["code_commit"] = sha
        return sha

    def gpu(self, require: bool = False) -> dict:
        """Report GPU status. With require=True, abort (exit 1) when there is no GPU: Kaggle may silently run
        on CPU if the account is unverified or the weekly quota is used up."""
        try:
            import torch
            ok = torch.cuda.is_available()
            info = {"torch": torch.__version__, "cuda_available": ok,
                    "device": torch.cuda.get_device_name(0) if ok else "cpu"}
        except Exception as e:  # torch missing
            info = {"torch": None, "cuda_available": False, "device": f"cpu ({type(e).__name__})"}
        self.env_info.update(info)
        self.progress(f"torch {info['torch']} | cuda available: {info['cuda_available']} | device: {info['device']}")
        if require and not info["cuda_available"]:
            self.error("No GPU available; aborting (required by this job).")
            self.write_env()
            raise SystemExit(1)
        return info

    def write_env(self, **extra) -> None:
        self.env_info.update(extra, finished=dt.datetime.now().isoformat())
        (self.out / "env.json").write_text(json.dumps(self.env_info, ensure_ascii=False), encoding="utf-8")

    def summary(self, obj: dict) -> None:
        (self.out / "summary.json").write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
        self.log("SUMMARY", json.dumps(obj, ensure_ascii=False))
        self.write_env()
