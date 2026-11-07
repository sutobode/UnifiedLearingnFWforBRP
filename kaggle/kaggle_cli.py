"""Local helper for running BRP jobs on Kaggle GPU and watching their logs.

Usage (run from the repo root):
    python kaggle/kaggle_cli.py auth                       # which credential is used (never prints secrets)
    python kaggle/kaggle_cli.py run   kaggle/jobs/<job>    # push -> stream live log -> download outputs
    python kaggle/kaggle_cli.py status <slug>               # QUEUED / RUNNING / COMPLETE / ERROR
    python kaggle/kaggle_cli.py logs   <slug> [-f]          # print log once, or follow it live
    python kaggle/kaggle_cli.py fetch  <slug>               # download outputs of the latest run

Where logs end up: kaggle_out/<slug>/<run_id>/  (run_id = local start time; LATEST.txt names the newest one).
Each `run` creates a NEW folder, so earlier logs are never overwritten. Files in that folder:
    live.log          every line streamed while the job runs, prefixed with local time
    status.log        status transitions (QUEUED -> RUNNING -> COMPLETE/ERROR) with timestamps
    <slug>.log        the full log file Kaggle produces (downloaded after the run)
    run.log           the job's own log file written to /kaggle/working (job-side, see job template)
    summary.json      final metrics written by the job
In the browser: https://www.kaggle.com/code/<username>/<slug>  (tab "Logs" of the running version)

Credentials, in order of precedence (handled by the Kaggle CLI itself):
    1. Environment: KAGGLE_API_TOKEN, or KAGGLE_USERNAME + KAGGLE_KEY
    2. A `.env` file at the repo root with the same variable names (loaded by this script; keep it out of git)
    3. ~/.kaggle/access_token (OAuth, from `python -m kaggle auth login`) or ~/.kaggle/kaggle.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "kaggle_out"
KAGGLE = [sys.executable, "-m", "kaggle"]
ENV_KEYS = ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY")
TERMINAL = ("complete", "error", "cancel")


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_env() -> dict:
    """Process env + optional repo-root .env (does not override variables already set)."""
    env = dict(os.environ)
    dotenv = REPO / ".env"
    if dotenv.exists():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k in ENV_KEYS and k not in env:
                env[k] = v
    env["PYTHONIOENCODING"] = "utf-8"
    return env


ENV = load_env()


def kaggle(*args: str, check: bool = True) -> str:
    p = subprocess.run([*KAGGLE, *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=ENV)
    out = (p.stdout or "") + (p.stderr or "")
    if check and p.returncode != 0:
        sys.exit(f"[kaggle_cli] command failed: kaggle {' '.join(args)}\n{out}")
    return out


def auth_source() -> str:
    if ENV.get("KAGGLE_API_TOKEN"):
        src = "env KAGGLE_API_TOKEN" + (" (from .env)" if "KAGGLE_API_TOKEN" not in os.environ else "")
    elif ENV.get("KAGGLE_USERNAME") and ENV.get("KAGGLE_KEY"):
        src = "env KAGGLE_USERNAME + KAGGLE_KEY" + (" (from .env)" if "KAGGLE_KEY" not in os.environ else "")
    else:
        home = Path.home() / ".kaggle"
        files = [f.name for f in (home / "access_token", home / "kaggle.json") if f.exists()]
        src = f"~/.kaggle ({', '.join(files) or 'NO credential file'})"
    return src


def username() -> str:
    for line in kaggle("config", "view", check=False).splitlines():
        if "username" in line:
            return line.split(":", 1)[1].strip()
    return ENV.get("KAGGLE_USERNAME", "")


def ref_of(slug_or_dir: str) -> tuple[str, str]:
    p = Path(slug_or_dir)
    if (p / "kernel-metadata.json").exists():
        ref = json.loads((p / "kernel-metadata.json").read_text(encoding="utf-8"))["id"]
    elif "/" in slug_or_dir:
        ref = slug_or_dir
    else:
        ref = f"{username()}/{slug_or_dir}"
    return ref, ref.split("/", 1)[1]


def status(ref: str) -> str:
    out = kaggle("kernels", "status", ref, check=False).strip()
    s = out.split('"')[-2] if '"' in out else out
    return s.replace("KernelWorkerStatus.", "").lower()


def follow(ref: str, out_dir: Path, poll: int, timeout: int) -> str:
    """Stream the live log into the console and live.log, while tracking status in status.log."""
    out_dir.mkdir(parents=True, exist_ok=True)
    live, stat_log = out_dir / "live.log", out_dir / "status.log"
    done = threading.Event()

    def stream():
        with live.open("a", encoding="utf-8") as f:
            while not done.is_set():
                p = subprocess.Popen([*KAGGLE, "kernels", "logs", ref, "-f"], stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                     errors="replace", env=ENV)
                for line in p.stdout:
                    line = f"[{now()}] {line.rstrip()}"
                    print(line, flush=True)
                    f.write(line + "\n"); f.flush()
                p.wait()
                done.wait(poll)  # stream ends while QUEUED or on hiccups: retry until finished

    t = threading.Thread(target=stream, daemon=True)
    t.start()
    last, start = None, time.time()
    with stat_log.open("a", encoding="utf-8") as f:
        while True:
            s = status(ref)
            if s != last:
                msg = f"[{now()}] STATUS {ref}: {s}"
                print(msg, flush=True); f.write(msg + "\n"); f.flush()
                last = s
            if any(k in s for k in TERMINAL):
                break
            if time.time() - start > timeout:
                print(f"[{now()}] local timeout after {timeout}s; job keeps running on Kaggle", flush=True)
                break
            time.sleep(poll)
    done.set()
    return last or ""


def fetch(ref: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", ref, "-p", str(out_dir), "-o", check=False)[-1500:])
    summ = out_dir / "summary.json"
    if summ.exists():
        print("[kaggle_cli] summary.json:\n" + summ.read_text(encoding="utf-8"))
        if '"gpu": false' in summ.read_text(encoding="utf-8").lower():
            print("[kaggle_cli] WARNING: job ran WITHOUT GPU (phone verification or quota?)")


# ---------------------------------------------------------------------------------------------------
# Code delivery + resume
#   * Jobs run the code that is ON GITHUB: main.py contains the placeholders @@CODE_REF@@ and @@REPO_URL@@,
#     replaced at push time by the local HEAD sha (which must already be pushed) and the origin URL.
#   * Resume: a job writes results.jsonl / state.json / checkpoints/ (see brp/resume.py). If state.json says
#     "finished": false, `run --auto-resume N` uploads the cumulative output as a PRIVATE dataset
#     (brp-resume-<slug>), attaches it to the next session, and repeats (at most N extra sessions).
# ---------------------------------------------------------------------------------------------------
def git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


def code_ref() -> tuple[str, str]:
    sha = git("rev-parse", "HEAD")
    url = git("remote", "get-url", "origin").removesuffix(".git")
    if not git("branch", "-r", "--contains", sha):
        sys.exit(f"[kaggle_cli] HEAD {sha[:8]} is not pushed to origin. Commit and `git push` first: "
                 "Kaggle runs the code from GitHub, not from this laptop.")
    return sha, url


def prepare_push_dir(job_dir: str, out_dir: Path, extra_datasets: list[str]) -> Path:
    sha, url = code_ref()
    dst = out_dir / "_push"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(job_dir, dst)
    mp = dst / "main.py"
    text = mp.read_text(encoding="utf-8").replace("@@CODE_REF@@", sha).replace("@@REPO_URL@@", url)
    mp.write_text(text, encoding="utf-8")
    meta = json.loads((dst / "kernel-metadata.json").read_text(encoding="utf-8"))
    for d in extra_datasets:
        if d not in meta["dataset_sources"]:
            meta["dataset_sources"].append(d)
    (dst / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "code_ref.txt").write_text(f"{url} @ {sha}\n", encoding="utf-8")
    print(f"[kaggle_cli] code: {url} @ {sha[:10]}  | datasets: {meta['dataset_sources'] or '-'}")
    return dst


def upload_resume_dataset(slug: str, prev_dir: Path) -> str:
    ds = f"brp-resume-{slug}"[:50].strip("-").lower()
    ref = f"{username()}/{ds}"
    pkg = OUT_ROOT / slug / "_resume_pkg"
    if pkg.exists():
        shutil.rmtree(pkg)
    pkg.mkdir(parents=True)
    for name in ("results.jsonl", "state.json"):
        if (prev_dir / name).exists():
            shutil.copy2(prev_dir / name, pkg / name)
    if (prev_dir / "checkpoints").exists():
        shutil.copytree(prev_dir / "checkpoints", pkg / "checkpoints")
    (pkg / "dataset-metadata.json").write_text(json.dumps(
        {"title": ds, "id": ref, "licenses": [{"name": "CC0-1.0"}]}), encoding="utf-8")
    common = ["-p", str(pkg), "-t", "-r", "zip"]
    exists = "ready" in kaggle("datasets", "status", ref, check=False).lower()
    out = kaggle("datasets", "version", *common, "-m", f"resume from {prev_dir.name}", check=False) if exists \
        else kaggle("datasets", "create", *common, check=False)
    print(f"[kaggle_cli] resume dataset {ref}: {out.strip()[-160:]}")
    for _ in range(60):
        if "ready" in kaggle("datasets", "status", ref, check=False).lower():
            return ref
        time.sleep(5)
    sys.exit(f"[kaggle_cli] dataset {ref} not ready after 5 minutes")


def read_state(run_dir: Path) -> dict | None:
    f = run_dir / "state.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def main() -> None:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("auth")
    r = sub.add_parser("run"); r.add_argument("job_dir")
    r.add_argument("--accelerator", default="NvidiaTeslaT4")
    r.add_argument("--poll", type=int, default=30); r.add_argument("--timeout", type=int, default=36000)
    r.add_argument("--resume-from", help="run_id (folder under kaggle_out/<slug>/) or LATEST to resume from")
    r.add_argument("--auto-resume", type=int, default=0, help="max extra sessions while state.json says unfinished")
    for name in ("status", "fetch"):
        sub.add_parser(name).add_argument("slug")
    lg = sub.add_parser("logs"); lg.add_argument("slug"); lg.add_argument("-f", "--follow", action="store_true")
    a = ap.parse_args()

    if a.cmd == "auth":
        print(f"credential source: {auth_source()}")
        print(f"username: {username() or '(unknown)'}")
        print(kaggle("kernels", "list", "--mine", "--page-size", "1", check=False)[:300])
        return

    ref, slug = ref_of(getattr(a, "job_dir", None) or a.slug)
    slug_dir = OUT_ROOT / slug
    latest_file = slug_dir / "LATEST.txt"
    if a.cmd == "run" or not latest_file.exists():
        # every `run` gets its own folder, so earlier logs are never overwritten
        out_dir = slug_dir / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        slug_dir.mkdir(parents=True, exist_ok=True)
        if a.cmd == "run":
            latest_file.write_text(out_dir.name, encoding="utf-8")
    else:
        out_dir = slug_dir / latest_file.read_text(encoding="utf-8").strip()
    print(f"[kaggle_cli] job {ref} | auth: {auth_source()} | logs -> {out_dir}")
    print(f"[kaggle_cli] browser: https://www.kaggle.com/code/{ref}")

    if a.cmd == "status":
        print(status(ref))
    elif a.cmd == "logs":
        if a.follow:
            follow(ref, out_dir, poll=15, timeout=10 ** 9)
        else:
            print(kaggle("kernels", "logs", ref, check=False))
    elif a.cmd == "fetch":
        fetch(ref, out_dir)
    elif a.cmd == "run":
        sessions, prev, extra = 0, None, []
        if a.resume_from:
            name = latest_file.read_text(encoding="utf-8").strip() if a.resume_from == "LATEST" else a.resume_from
            prev = slug_dir / name
        while True:
            if prev is not None:
                extra = [upload_resume_dataset(slug, prev)]
            push_dir = prepare_push_dir(a.job_dir, out_dir, extra)
            push = ["kernels", "push", "-p", str(push_dir)]
            if a.accelerator.lower() != "none":
                push += ["--accelerator", a.accelerator]
            print(kaggle(*push)[-800:])
            final = follow(ref, out_dir, a.poll, a.timeout)
            fetch(ref, out_dir)
            if "error" in final:
                sys.exit(f"[kaggle_cli] job ended with status {final}; see {out_dir / 'live.log'}")
            st = read_state(out_dir)
            if st is None or st.get("finished", True) or sessions >= a.auto_resume:
                if st and not st.get("finished", True):
                    print(f"[kaggle_cli] UNFINISHED ({st}); continue with: run ... --resume-from {out_dir.name}")
                break
            sessions += 1
            print(f"[kaggle_cli] unfinished ({st.get('n_done')} done); auto-resume session {sessions}/{a.auto_resume}")
            prev = out_dir
            out_dir = slug_dir / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            latest_file.write_text(out_dir.name, encoding="utf-8")

if __name__ == "__main__":
    main()
