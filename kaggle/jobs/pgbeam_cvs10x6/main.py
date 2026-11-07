"""PG-Beam vs L2BR best-of-M on CVS uBRP H=10 S=6 (40 public instances), public L2BR SIL model, T4 GPU.
L2BR reference (job brp-g0-headroom-cvs10x6, temperature 1, original code): M=256 -> 66.150 (1.4 s/inst),
M=2560 -> 65.275 (6.5 s/inst). Resumable: every (config, instance) is one JSONL row."""
import os, subprocess, sys, time, traceback, json

REPO_URL, CODE_REF, CODE = "@@REPO_URL@@", "@@CODE_REF@@", "/kaggle/temp/code"
L2BR_URL, L2BR_COMMIT, L2BR = "https://github.com/operagang/L2BR", "941ce86afc09cf4e1c62400b9c5d17407a82d297", "/kaggle/temp/L2BR"
os.makedirs("/kaggle/temp", exist_ok=True)
for url, path, ref in ((REPO_URL, CODE, CODE_REF), (L2BR_URL, L2BR, L2BR_COMMIT)):
    subprocess.run(["git", "clone", "-q", url, path], check=True)
    subprocess.run(["git", "-C", path, "checkout", "-q", ref], check=True)
sys.path.insert(0, CODE)

from statistics import mean
from brp.data import cvs_tiers, load_cvs
from brp.joblog import JobLog
from brp.l2br_adapter import load_model, log_probs
from brp.resume import ResumableRun, find_resume_dirs
from brp.search.pg_beam import pg_beam
from brp.sim import replay

OUT, H, S, VARIANT = "/kaggle/working", 10, 6, "u"
T = cvs_tiers(H)
GRID = [(B, 3, lam) for B in (16, 64, 256) for lam in (0.0, 1.0, 4.0)]   # (B, k, lambda)
REF = {"L2BR M=256": (66.150, 1.38), "L2BR M=2560": (65.275, 6.54)}
PER_INSTANCE_LIMIT_S = 60


def main():
    jl = JobLog(OUT)
    info = jl.gpu(require=True)
    jl.progress(f"code commit {jl.git_head(CODE)[:10]} (pinned {CODE_REF[:10]}), L2BR {L2BR_COMMIT[:10]}")
    device = "cuda"
    model = load_model(f"{L2BR}/SIL/final_models/CVS-uBRP/H={H}xS={S}.pt", VARIANT, device, L2BR)
    insts = load_cvs(H, S, L2BR)
    jl.progress(f"{len(insts)} instances CVS H={H} S={S} (T={T}); configs: {len(GRID)}")
    run = ResumableRun(OUT, resume_dirs=find_resume_dirs(), budget_s=9 * 3600, margin_s=600)
    jl.progress(f"inherited rows: {run.n_inherited}")
    policy = lambda states: log_probs(model, states, T, VARIANT, device)

    items = [(f"B{B}_k{k}_lam{lam}/inst{i:02d}", (B, k, lam, i)) for (B, k, lam) in GRID for i in range(len(insts))]

    def work(p):
        B, k, lam, i = p
        r = pg_beam(policy, insts[i], T, VARIANT, B=B, k=k, lam=lam, time_limit_s=PER_INSTANCE_LIMIT_S)
        ok = r.obj is not None and replay(insts[i], T, VARIANT, r.path) == r.obj   # re-check every solution
        return {"B": B, "k": k, "lam": lam, "inst": i, "obj": r.obj if ok else None, "valid": ok,
                "wall_s": round(r.wall_s, 3), "nn_calls": r.nn_calls, "attempts": r.stats.get("attempts"),
                "timed_out": r.timed_out}

    finished = run.run_items(items, work, log=jl.progress)

    rows = [json.loads(l) for l in open(f"{OUT}/results.jsonl", encoding="utf-8")]
    summary = {"finished": finished, "device": info["device"], "ref": REF, "configs": []}
    for (B, k, lam) in GRID:
        rs = [r for r in rows if (r["B"], r["k"], r["lam"]) == (B, k, lam)]
        if len(rs) == len(insts):
            c = {"B": B, "k": k, "lam": lam, "mean_obj": round(mean(r["obj"] for r in rs if r["valid"]), 3),
                 "mean_s": round(mean(r["wall_s"] for r in rs), 2), "invalid": sum(not r["valid"] for r in rs),
                 "timeouts": sum(bool(r["timed_out"]) for r in rs), "widened": sum((r["attempts"] or 1) > 1 for r in rs)}
            summary["configs"].append(c)
            jl.result(c)
    summary["finished"] = finished
    jl.summary(summary)


try:
    main()
except SystemExit:
    raise
except Exception:
    print("[ERROR] " + traceback.format_exc(), flush=True)
    raise
