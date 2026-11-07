"""Self-test of the resume mechanism (CPU only). 9 items x 15 s with a 55 s budget per session, so the job
MUST stop unfinished after ~3 items and the driver (kaggle_cli.py run --auto-resume) has to resume it twice.
Code is fetched from GitHub at the pinned commit (placeholders replaced by kaggle_cli.py at push time)."""
import os, subprocess, sys, time, traceback

REPO_URL, CODE_REF, CODE = "@@REPO_URL@@", "@@CODE_REF@@", "/kaggle/temp/code"
os.makedirs("/kaggle/temp", exist_ok=True)
subprocess.run(["git", "clone", "-q", REPO_URL, CODE], check=True)
subprocess.run(["git", "-C", CODE, "checkout", "-q", CODE_REF], check=True)
sys.path.insert(0, CODE)

from brp.joblog import JobLog
from brp.resume import ResumableRun, find_resume_dirs

OUT = "/kaggle/working"


def main():
    jl = JobLog(OUT)
    jl.gpu(require=False)
    jl.progress(f"code commit {jl.git_head(CODE)[:10]} (pinned {CODE_REF[:10]})")
    run = ResumableRun(OUT, resume_dirs=find_resume_dirs(), budget_s=55, margin_s=5)
    jl.progress(f"resume dirs: {find_resume_dirs()} | inherited rows: {run.n_inherited}")

    def work(i):
        time.sleep(15)
        jl.progress(f"item {i} done")
        return {"val": i * i}

    finished = run.run_items([(f"item{i:02d}", i) for i in range(9)], work, log=jl.progress)
    jl.result({"finished": finished, "n_done": len(run.done), "n_inherited": run.n_inherited})
    jl.summary({"finished": finished, "n_done": len(run.done), "n_inherited": run.n_inherited})


try:
    main()
except SystemExit:
    raise
except Exception:
    print("[ERROR] " + traceback.format_exc(), flush=True)
    raise
