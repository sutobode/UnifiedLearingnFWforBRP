"""Statistics for result files (JSONL rows with at least: cfg, inst, obj, wall_s).

    python -m brp.stats results.jsonl [--ref "<cfg>"]      # <= ~30 lines: per-config means + paired test vs ref
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from statistics import mean

import numpy as np
from scipy import stats as sps


def load_rows(*paths) -> list:
    rows = []
    for p in paths:
        for ln in open(p, encoding="utf-8"):
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    return rows


def by_cfg(rows) -> dict:
    out = defaultdict(dict)
    for r in rows:
        if r.get("obj") is not None and r.get("valid", True):
            out[r["cfg"]][r["inst"]] = r
    return out


def paired(a: dict, b: dict, field: str = "obj", n_boot: int = 10000, seed: int = 0) -> dict:
    """a, b: inst -> row. d = a - b on shared instances (negative = a uses fewer relocations)."""
    keys = sorted(set(a) & set(b), key=str)
    d = np.array([a[k][field] - b[k][field] for k in keys], dtype=float)
    out = {"n": len(d), "mean_diff": float(d.mean()) if len(d) else float("nan"),
           "wins": int((d < 0).sum()), "ties": int((d == 0).sum()), "losses": int((d > 0).sum())}
    if len(d):
        boots = np.random.default_rng(seed).choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
        out["ci95"] = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
        out["p_wilcoxon"] = float(sps.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    return out


def table(rows, ref: str | None = None) -> str:
    groups = by_cfg(rows)
    head = f"{'config':34s} {'n':>4s} {'mean obj':>9s} {'mean s':>8s}"
    lines = [head + (f" {'d vs ref':>9s} {'95% CI':>17s} {'p':>7s} W/T/L" if ref else "")]
    for cfg in sorted(groups, key=lambda c: mean(r["obj"] for r in groups[c].values())):
        g = groups[cfg]
        line = f"{cfg:34s} {len(g):4d} {mean(r['obj'] for r in g.values()):9.3f} {mean(r['wall_s'] for r in g.values()):8.2f}"
        if ref and ref in groups and cfg != ref:
            p = paired(g, groups[ref])
            line += (f" {p['mean_diff']:+9.3f} [{p['ci95'][0]:+.3f},{p['ci95'][1]:+.3f}]"
                     f" {p['p_wilcoxon']:7.4f} {p['wins']}/{p['ties']}/{p['losses']}")
        lines.append(line)
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--ref")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(table(load_rows(*a.paths), a.ref))
