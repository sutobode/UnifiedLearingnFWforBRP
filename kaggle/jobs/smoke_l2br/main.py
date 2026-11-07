"""Kaggle smoke test: GPU check + reproduce L2BR-Base numbers (Table 2, CVS-uBRP, H=3).

Expected (paper, Table 2, L2BR Base column): S=3 -> 4.975, S=4 -> 6.025, S=5 -> 6.850.

Log files written to /kaggle/working (downloaded to kaggle_out/<slug>/ after the run):
    run.log         every line printed, timestamped, prefixed [PROGRESS]/[RESULT]/[SUMMARY]/[WARN]/[ERROR]
    results.jsonl   one JSON line per (H, S) group
    summary.json    final summary
    env.json        git commit, torch/python versions, GPU, start/end time
"""
import datetime as dt
import json, os, platform, re, subprocess, sys, time

L2BR_URL = "https://github.com/operagang/L2BR"
L2BR_COMMIT = "941ce86afc09cf4e1c62400b9c5d17407a82d297"
WORK = "/kaggle/working"
REPO = "/kaggle/temp/L2BR"
EXPECTED = {3: 4.975, 4: 6.025, 5: 6.850}
os.makedirs(WORK, exist_ok=True)
RUN_LOG = open(os.path.join(WORK, "run.log"), "a", encoding="utf-8")


def log(tag, msg):
    """Print + append to run.log, flushed immediately so `kernels logs -f` shows it live."""
    line = f"[{dt.datetime.now().strftime('%H:%M:%S')}] [{tag}] {msg}"
    print(line, flush=True)
    RUN_LOG.write(line + "\n")
    RUN_LOG.flush()


def sh(cmd, cwd=None):
    log("PROGRESS", "$ " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    for stream in (r.stdout, r.stderr):
        for ln in (stream or "").strip().splitlines()[-30:]:
            log("PROGRESS", "  " + ln)
    return r


def write_json(name, obj, mode="w"):
    with open(os.path.join(WORK, name), mode, encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + ("\n" if mode == "a" else ""))


def main():
    t_start = time.time()
    import torch
    gpu = torch.cuda.is_available()
    device = torch.cuda.get_device_name(0) if gpu else "cpu"
    env = {"started": dt.datetime.now().isoformat(), "python": platform.python_version(),
           "torch": torch.__version__, "cuda_available": gpu, "device": device}
    log("PROGRESS", f"torch {torch.__version__} | cuda available: {gpu} | device: {device}")
    summary = {"gpu": gpu, "device": device, "runs": []}
    if not gpu:
        log("ERROR", "No GPU (account not phone-verified or quota exhausted). L2BR code requires CUDA.")
        write_json("env.json", env); write_json("summary.json", summary)
        sys.exit(1)

    os.makedirs("/kaggle/temp", exist_ok=True)
    if sh(["git", "clone", "-q", L2BR_URL, REPO]).returncode != 0:
        log("ERROR", "git clone failed (internet disabled?)")
        sys.exit(1)
    sh(["git", "checkout", "-q", L2BR_COMMIT], cwd=REPO)
    head = sh(["git", "rev-parse", "HEAD"], cwd=REPO).stdout.strip()
    env["l2br_commit"] = head
    summary["l2br_commit"] = head
    log("PROGRESS", f"L2BR commit {head} (pinned {L2BR_COMMIT}) match={head == L2BR_COMMIT}")

    # Linux fix: L2BR's CVS loader hard-codes a Windows path ('datasets\\CVS'). Patch it in the clone.
    loader = os.path.join(REPO, "benchmark_loader.py")
    src = open(loader, encoding="utf-8").read()
    old_path, new_path = r"'datasets\\CVS'", "'datasets/CVS'"
    if old_path in src:
        open(loader, "w", encoding="utf-8").write(src.replace(old_path, new_path))
        log("PROGRESS", "patched benchmark_loader.py: datasets\\CVS -> datasets/CVS")
    else:
        log("WARN", "Windows-path pattern not found in benchmark_loader.py (already fixed or code changed)")

    for i, (S, exp) in enumerate(EXPECTED.items(), 1):
        log("PROGRESS", f"group {i}/{len(EXPECTED)}: CVS-uBRP H=3 S={S} (T=5, N=2560 samples)")
        t0 = time.time()
        r = sh([sys.executable, "test.py", "-pt", "uBRP", "-bt", "CVS", "-t", "5", "-s", str(S),
                "-mp", "./IL/final_models/CVS-uBRP/H=5xS=5.pt", "-N", "2560", "-T", "1"], cwd=REPO)
        m = re.search(r"Test Score:\s*([0-9.]+)", r.stdout)
        score = float(m.group(1)) if m else None
        row = {"H": 3, "S": S, "score": score, "expected": exp,
               "rel_diff_pct": None if score is None else round(100 * (score - exp) / exp, 3),
               "wall_s": round(time.time() - t0, 1)}
        summary["runs"].append(row)
        write_json("results.jsonl", row, mode="a")
        log("RESULT", json.dumps(row))
        if score is None:
            log("WARN", "no 'Test Score' found in test.py output; see lines above")

    env["finished"] = dt.datetime.now().isoformat()
    env["wall_s"] = round(time.time() - t_start, 1)
    write_json("env.json", env)
    write_json("summary.json", summary)
    log("SUMMARY", json.dumps(summary))


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:  # make the failure visible in run.log, not just in Kaggle's log
        import traceback
        log("ERROR", f"{type(e).__name__}: {e}")
        for ln in traceback.format_exc().splitlines():
            log("ERROR", ln)
        raise
