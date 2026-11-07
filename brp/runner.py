"""Resumable experiment loops shared by Kaggle jobs: PG-Beam grids and per-instance L2BR best-of-M baselines.
Every row: {key, cfg, inst, obj, valid, wall_s, ...}; one JSONL line per (config, instance)."""
from __future__ import annotations

import time

from .l2br_sampler import l2br_best_of_m
from .search.pg_beam import pg_beam
from .sim import replay


def cfg_name(B, k, lam) -> str:
    return f"pgbeam B{B} k{k} lam{lam}"


def l2br_name(M, temp) -> str:
    return f"l2br M{M} T{temp}"


def run_pgbeam(jl, run, policy, items, T, variant, grid, limit_s=60.0):
    """grid: [(B, k, lam)]. items: [(inst_key, stacks)]. Returns True when everything is done."""
    todo = [(f"{cfg_name(B, k, lam)}/{ik}", (B, k, lam, ik, st)) for (B, k, lam) in grid for ik, st in items]

    def work(p):
        B, k, lam, ik, st = p
        r = pg_beam(policy, st, T, variant, B=B, k=k, lam=lam, time_limit_s=limit_s)
        ok = r.obj is not None and replay(st, T, variant, r.path) == r.obj      # re-check every solution
        return {"cfg": cfg_name(B, k, lam), "inst": ik, "obj": r.obj if ok else None, "valid": ok,
                "wall_s": round(r.wall_s, 3), "nn_calls": r.nn_calls, "attempts": r.stats.get("attempts"),
                "timed_out": r.timed_out}
    return run.run_items(todo, work, log=jl.progress)


def run_l2br(jl, run, model, items, T, variant, Ms, temp=1.0, device="cpu", traj_limit=500):
    """Per-instance L2BR best-of-M baseline (for paired comparison with run_pgbeam)."""
    todo = [(f"{l2br_name(M, temp)}/{ik}", (M, ik, st)) for M in Ms for ik, st in items]

    def work(p):
        M, ik, st = p
        t0 = time.time()
        obj = l2br_best_of_m(model, st, T, variant, M, temp, device, traj_limit)
        return {"cfg": l2br_name(M, temp), "inst": ik, "obj": obj, "valid": obj is not None,
                "wall_s": round(time.time() - t0, 3)}
    return run.run_items(todo, work, log=jl.progress)
