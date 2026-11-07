"""Paired comparisons between configs of one results.jsonl (mean obj, mean time, diff, 95% CI, Wilcoxon, W/T/L).

    python scripts/compare_pairs.py <slug> "<cfg A>" "<cfg B>" ["<cfg A2>" "<cfg B2>" ...]

Reads kaggle_out/<slug>/<LATEST run>/results.jsonl. d = A - B on shared instances (negative = A better).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brp.stats import by_cfg, load_rows, paired  # noqa: E402


def main():
    slug, cfgs = sys.argv[1], sys.argv[2:]
    base = Path(__file__).resolve().parents[1] / "kaggle_out" / slug
    run = (base / "LATEST.txt").read_text().strip()
    rows = load_rows(base / run / "results.jsonl")
    g = by_cfg(rows)
    print(f"run {run} | rows {len(rows)} | invalid {sum(1 for r in rows if not r.get('valid', True))} | "
          f"timeouts {sum(bool(r.get('timed_out')) for r in rows)}")
    mean = lambda c, f: sum(r[f] for r in g[c].values()) / len(g[c])
    for a, b in zip(cfgs[::2], cfgs[1::2]):
        p = paired(g[a], g[b])
        pct = 100 * p["mean_diff"] / mean(b, "obj")
        print(f"{a} vs {b}\n   obj {mean(a, 'obj'):.3f} vs {mean(b, 'obj'):.3f} | time {mean(a, 'wall_s'):.2f}s vs "
              f"{mean(b, 'wall_s'):.2f}s | diff {p['mean_diff']:+.3f} ({pct:+.2f}%) CI [{p['ci95'][0]:+.3f},"
              f"{p['ci95'][1]:+.3f}] p={p['p_wilcoxon']:.4f} W/T/L {p['wins']}/{p['ties']}/{p['losses']}")


if __name__ == "__main__":
    main()
