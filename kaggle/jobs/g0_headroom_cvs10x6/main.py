"""G0 headroom test (task T1.1): does L2BR's best-of-M keep improving as M grows?

Public L2BR model, CVS uBRP, H=10 (T=12), S=6, 40 instances, temperature 1.
Mean #relocations for M in 256 / 2560 / 25600. Paper reference (Table 2, H=10, S=6):
L2BR-Large 66.700, L2BR-Targeted 65.200 (paper picks best temperature per scale; we use T=1).
Decision rule (FINAL proposal, gate G0): headroom exists if mean obj drops >= 0.3% from M=2560 to M=25600.

Logs in /kaggle/working: run.log, results.jsonl, summary.json, env.json  (tags: [PROGRESS] [RESULT] [SUMMARY] [WARN] [ERROR]).
"""
import datetime as dt
import json, os, platform, re, subprocess, sys, time

L2BR_URL = "https://github.com/operagang/L2BR"
L2BR_COMMIT = "941ce86afc09cf4e1c62400b9c5d17407a82d297"
WORK, REPO = "/kaggle/working", "/kaggle/temp/L2BR"
MODEL = "./SIL/final_models/CVS-uBRP/H=10xS=6.pt"
M_LIST = [256, 2560, 25600]
os.makedirs(WORK, exist_ok=True)
RUN_LOG = open(os.path.join(WORK, "run.log"), "a", encoding="utf-8")


def log(tag, msg):
    line = f"[{dt.datetime.now().strftime('%H:%M:%S')}] [{tag}] {msg}"
    print(line, flush=True)
    RUN_LOG.write(line + "\n"); RUN_LOG.flush()


def sh(cmd, cwd=None, quiet=False):
    log("PROGRESS", "$ " + " ".join(cmd))
    env = dict(os.environ, TQDM_DISABLE="1", PYTHONUNBUFFERED="1")
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    if not quiet or r.returncode != 0:
        for stream in (r.stdout, r.stderr):
            for ln in (stream or "").strip().splitlines()[-15:]:
                log("PROGRESS", "  " + ln)
    return r


def dump(name, obj, mode="w"):
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
    summary = {"gpu": gpu, "device": device, "task": "G0 headroom CVS-uBRP H=10 S=6", "runs": []}
    if not gpu:
        log("ERROR", "No GPU available; aborting.")
        dump("env.json", env); dump("summary.json", summary); sys.exit(1)

    os.makedirs("/kaggle/temp", exist_ok=True)
    if sh(["git", "clone", "-q", L2BR_URL, REPO], quiet=True).returncode != 0:
        log("ERROR", "git clone failed"); sys.exit(1)
    sh(["git", "checkout", "-q", L2BR_COMMIT], cwd=REPO, quiet=True)
    head = sh(["git", "rev-parse", "HEAD"], cwd=REPO, quiet=True).stdout.strip()
    env["l2br_commit"] = head
    log("PROGRESS", f"L2BR commit match={head == L2BR_COMMIT}")
    loader = os.path.join(REPO, "benchmark_loader.py")
    src = open(loader, encoding="utf-8").read()
    if r"'datasets\\CVS'" in src:
        open(loader, "w", encoding="utf-8").write(src.replace(r"'datasets\\CVS'", "'datasets/CVS'"))
        log("PROGRESS", "patched benchmark_loader.py: datasets\\CVS -> datasets/CVS")

    for M in M_LIST:
        t0 = time.time()
        log("PROGRESS", f"M={M}: start")
        r = sh([sys.executable, "test.py", "-pt", "uBRP", "-bt", "CVS", "-t", "12", "-s", "6",
                "-mp", MODEL, "-N", str(M), "-T", "1"], cwd=REPO, quiet=True)
        m = re.search(r"Test Score:\s*([0-9.]+)", r.stdout)
        row = {"H": 10, "S": 6, "M": M, "temp": 1, "mean_obj": float(m.group(1)) if m else None,
               "wall_s": round(time.time() - t0, 1), "paper_large": 66.700, "paper_targeted": 65.200}
        summary["runs"].append(row)
        dump("results.jsonl", row, mode="a")
        log("RESULT", json.dumps(row))
        if not m:
            log("WARN", "no Test Score; stopping the M sweep"); break

    objs = {r["M"]: r["mean_obj"] for r in summary["runs"] if r["mean_obj"] is not None}
    if 2560 in objs and 25600 in objs:
        drop = 100 * (objs[2560] - objs[25600]) / objs[2560]
        summary["drop_pct_2560_to_25600"] = round(drop, 3)
        summary["g0_headroom"] = bool(drop >= 0.3)
        log("RESULT", f"drop 2560->25600 = {drop:.3f}%  (G0 threshold 0.3%)  headroom={drop >= 0.3}")
    env["finished"] = dt.datetime.now().isoformat(); env["wall_s"] = round(time.time() - t_start, 1)
    dump("env.json", env); dump("summary.json", summary)
    log("SUMMARY", json.dumps(summary))


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        import traceback
        log("ERROR", f"{type(e).__name__}: {e}")
        for ln in traceback.format_exc().splitlines():
            log("ERROR", ln)
        raise
