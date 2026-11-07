"""Compact digest of one Kaggle run, to read results WITHOUT loading whole logs into an agent's context.

    python kaggle/digest.py <slug> [--run RUN_ID] [--tail 5]

Prints ~20 lines: status timeline, GPU line, [RESULT]/[SUMMARY] lines, error/warn counts + first errors,
and the last few lines of run.log. Reads kaggle_out/<slug>/<run_id>/ (newest by default, via LATEST.txt;
falls back to a flat kaggle_out/<slug>/ folder).
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "kaggle_out"


def cut(s, n=200):
    s = s.rstrip()
    return s if len(s) <= n else s[: n - 3] + "..."


def lines(p):
    return p.read_text(encoding="utf-8", errors="replace").splitlines() if p.exists() else []


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("--run")
    ap.add_argument("--tail", type=int, default=5)
    a = ap.parse_args()

    base = ROOT / a.slug
    if a.run:
        d = base / a.run
    elif (base / "LATEST.txt").exists():
        d = base / (base / "LATEST.txt").read_text(encoding="utf-8").strip()
    else:
        d = base
    if not d.exists():
        sys.exit(f"no such run folder: {d}")
    print(f"RUN {d.relative_to(ROOT.parent)}")

    # status timeline (collapse consecutive duplicates, shorten long messages)
    last, tl = None, []
    for ln in lines(d / "status.log"):
        m = re.search(r"STATUS \S+: (.*)", ln)
        if m:
            st = cut(m.group(1), 60)
            if st != last:
                tl.append(f"{ln[1:9] if ln.startswith('[') else ''}{ln[9:20] if ln.startswith('[') else ''} {st}".strip())
                last = st
    print("STATUS " + " | ".join(tl) if tl else "STATUS (no status.log)")

    run = lines(d / "run.log") or lines(d / "live.log")
    gpu = next((l for l in run if "cuda available" in l), None)
    print("GPU    " + (cut(gpu.split("] ", 2)[-1], 120) if gpu else "(no GPU line found)"))

    results = [l for l in run if "[RESULT]" in l]
    for l in results[:15]:
        print("RESULT " + cut(l.split("[RESULT] ", 1)[-1], 200))
    if len(results) > 15:
        print(f"RESULT ... +{len(results) - 15} more (see results.jsonl)")
    summ = [l for l in run if "[SUMMARY]" in l]
    if summ:
        print("SUMMARY " + cut(summ[-1].split("[SUMMARY] ", 1)[-1], 400))

    errs = [l for l in run if "[ERROR]" in l or "Traceback" in l]
    warns = [l for l in run if "[WARN]" in l]
    print(f"ERRORS {len(errs)}  WARNS {len(warns)}")
    for l in errs[:5]:
        print("  E " + cut(l.split("] ", 2)[-1], 200))
    for l in warns[:3]:
        print("  W " + cut(l.split("] ", 2)[-1], 200))
    if a.tail:
        print(f"TAIL (run.log, last {a.tail}, progress bars hidden)")
        for l in [x for x in run if "it/s]" not in x and "s/it]" not in x][-a.tail:]:
            print("  " + cut(l, 160))
    files = sorted(p.name for p in d.iterdir() if p.is_file())
    print("FILES " + ", ".join(files))


if __name__ == "__main__":
    main()
