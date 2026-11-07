"""T2.3: GPU PG-Beam vs per-instance L2BR best-of-M on several public benchmark groups (fixed configs, no per-group
tuning: k=3, lambda=1, B in 256/1024/4096). Resumable; one JSONL row per (group, config, instance)."""
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

OUT = "/kaggle/working"
# (benchmark, variant, a, S, subsample step): a = H for CVS, T for ZQLZ; step>1 keeps every step-th instance
GROUPS = [("CVS", "r", 10, 6, 1), ("CVS", "u", 5, 10, 1), ("CVS", "u", 6, 6, 1), ("CVS", "u", 6, 10, 1),
          ("ZQLZ", "u", 7, 10, 10), ("CVS", "u", 10, 10, 1)]
GRID = [(256, 3, 1.0), (1024, 3, 1.0), (4096, 3, 1.0)]
MS = (256, 2560)


def gname(g):
    b, v, a, S, _ = g
    return f"{b}-{v}BRP-{'H' if b == 'CVS' else 'T'}{a}S{S}"


def main():
    jl = JobLog(OUT)
    info = jl.gpu(require=True)
    jl.progress(f"code commit {jl.git_head(CODE)[:10]} (pinned {CODE_REF[:10]}), L2BR {L2BR_COMMIT[:10]}")
    run = ResumableRun(OUT, resume_dirs=find_resume_dirs(), budget_s=9 * 3600, margin_s=900)
    jl.progress(f"inherited rows: {run.n_inherited}; groups: {[gname(g) for g in GROUPS]}")
    finished = True
    for g in GROUPS:
        b, v, a, S, step = g
        if run.should_stop():
            finished = False
            break
        items, T = load_benchmark(b, a, S, L2BR)
        items = items[::step]
        model = load_model(l2br_model_path(b, v, a, S, L2BR), v, "cuda", L2BR)
        tag = gname(g)
        jl.progress(f"== group {tag}: {len(items)} instances, T={T}")
        ok1 = run_l2br(jl, run, model, items, T, v, Ms=MS, temp=1.0, device="cuda", tag=tag)
        ok2 = run_pgbeam(jl, run, gpu_search(lambda x: model(x), T, v, "cuda"), items, T, v, GRID,
                         limit_s=180, tag=f"{tag} gpubeam")
        finished = finished and ok1 and ok2
        del model
    rows = load_rows(f"{OUT}/results.jsonl")
    for g in GROUPS:
        pre = gname(g)
        sub = [r for r in rows if r["cfg"].startswith(pre)]
        if sub:
            jl.result({"group": pre})
            for line in table(sub, ref=f"{pre} l2br M2560 T1.0").splitlines():
                jl.result({"table": line})
    jl.summary({"finished": finished, "device": info["device"], "n_rows": len(rows)})


try:
    main()
except SystemExit:
    raise
except Exception:
    print("[ERROR] " + traceback.format_exc(), flush=True)
    raise
