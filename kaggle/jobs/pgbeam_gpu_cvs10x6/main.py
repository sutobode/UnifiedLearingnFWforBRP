"""Vectorised GPU PG-Beam (large beams) vs per-instance L2BR best-of-M, CVS uBRP H=10 S=6 (40 public instances).
Reference (job brp-g0-headroom-cvs10x6, original L2BR code, T=1): M=256 -> 66.150 (1.4 s), M=2560 -> 65.275
(6.5 s), M=25600 -> 64.700 (59.5 s). Resumable. Python-beam result (brp-pgbeam-cvs10x6): B256 -> 65.600 (7.9 s)."""
import os, subprocess, sys, traceback

REPO_URL, CODE_REF, CODE = "@@REPO_URL@@", "@@CODE_REF@@", "/kaggle/temp/code"
L2BR_URL, L2BR_COMMIT, L2BR = "https://github.com/operagang/L2BR", "941ce86afc09cf4e1c62400b9c5d17407a82d297", "/kaggle/temp/L2BR"
os.makedirs("/kaggle/temp", exist_ok=True)
for url, path, ref in ((REPO_URL, CODE, CODE_REF), (L2BR_URL, L2BR, L2BR_COMMIT)):
    subprocess.run(["git", "clone", "-q", url, path], check=True)
    subprocess.run(["git", "-C", path, "checkout", "-q", ref], check=True)
sys.path.insert(0, CODE)

from brp.benchmarks import l2br_model_path, load_benchmark
from brp.joblog import JobLog
from brp.l2br_adapter import load_model
from brp.resume import ResumableRun, find_resume_dirs
from brp.runner import gpu_search, run_l2br, run_pgbeam
from brp.stats import load_rows, table

OUT, VARIANT, H, S = "/kaggle/working", "u", 10, 6
GRID = [(256, 3, 1.0), (1024, 3, 1.0), (4096, 3, 1.0), (16384, 3, 1.0),
        (1024, 3, 0.5), (1024, 3, 2.0), (1024, 2, 1.0), (1024, 5, 1.0)]       # (B, k, lambda)


def main():
    jl = JobLog(OUT)
    info = jl.gpu(require=True)
    jl.progress(f"code commit {jl.git_head(CODE)[:10]} (pinned {CODE_REF[:10]}), L2BR {L2BR_COMMIT[:10]}")
    items, T = load_benchmark("CVS", H, S, L2BR)
    model = load_model(l2br_model_path("CVS", VARIANT, H, S, L2BR), VARIANT, "cuda", L2BR)
    jl.progress(f"{len(items)} instances CVS H={H} S={S} (T={T}); {len(GRID)} PG-Beam configs")
    run = ResumableRun(OUT, resume_dirs=find_resume_dirs(), budget_s=9 * 3600, margin_s=600)
    jl.progress(f"inherited rows: {run.n_inherited}")
    ok1 = run_l2br(jl, run, model, items, T, VARIANT, Ms=(256, 2560), temp=1.0, device="cuda")
    ok2 = run_pgbeam(jl, run, gpu_search(lambda x: model(x), T, VARIANT, "cuda"), items, T, VARIANT, GRID,
                     limit_s=120, tag="gpubeam")
    finished = ok1 and ok2
    rows = load_rows(f"{OUT}/results.jsonl")
    for line in table(rows, ref="l2br M2560 T1.0").splitlines():
        jl.result({"table": line})
    jl.summary({"finished": finished, "device": info["device"], "n_rows": len(rows)})


try:
    main()
except SystemExit:
    raise
except Exception:
    print("[ERROR] " + traceback.format_exc(), flush=True)
    raise
