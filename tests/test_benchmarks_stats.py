import random
from pathlib import Path

import numpy as np
import pytest

from brp.benchmarks import l2br_model_path, load_benchmark
from brp.data import DEFAULT_L2BR, random_instance
from brp.exact import solve
from brp.stats import paired, table

ROOT = Path(DEFAULT_L2BR)
needs_l2br = pytest.mark.skipif(not (ROOT / "datasets").exists(), reason="third_party/L2BR missing")


@needs_l2br
def test_cvs_all_groups_have_forty_valid_instances():
    items, T = load_benchmark("CVS", 5, 6)
    assert T == 7 and len(items) == 40
    assert all(sum(map(len, st)) == 30 and max(map(len, st)) <= T for _, st in items)


@needs_l2br
def test_zqlz_counts_and_validity():
    items, T = load_benchmark("ZQLZ", 5, 8)
    assert T == 5 and len(items) == 500          # N in {T*S-T .. T*S-1} = 35..39: 5 folders x 100 instances
    for _, st in items:
        flat = sorted(p for s in st for p in s)
        assert flat == list(range(1, len(flat) + 1))   # priorities are exactly 1..N
        assert len(st) == 8 and max(map(len, st)) <= T


@needs_l2br
def test_ll_r_and_u_counts():
    r, T = load_benchmark("LL-R", 6, 16)
    u, _ = load_benchmark("LL-U", 6, 16)
    assert T == 6 and len(r) == 5 and len(u) == 2  # 5 random + 2 upside-down instances per group (paper Sec. 4.1)
    for _, st in r + u:
        assert len(st) == 16 and max(map(len, st)) <= 6
        flat = sorted(p for s in st for p in s)
        assert flat == list(range(1, len(flat) + 1))


@needs_l2br
def test_every_listed_model_path_exists_for_cvs_h10():
    assert l2br_model_path("CVS", "u", 10, 6).exists()
    assert l2br_model_path("ZQLZ", "u", 7, 10).exists()
    assert l2br_model_path("LL-R", "r", 6, 16).exists()


def rows(cfg, objs, t=1.0):
    return [{"cfg": cfg, "inst": i, "obj": o, "valid": True, "wall_s": t} for i, o in enumerate(objs)]


def test_paired_stats_known_values():
    a = {i: {"obj": 10 - 1} for i in range(20)}       # 9 everywhere
    b = {i: {"obj": 10} for i in range(20)}           # 10 everywhere
    p = paired(a, b)
    assert p["mean_diff"] == -1 and p["wins"] == 20 and p["losses"] == 0
    assert p["ci95"] == (-1.0, -1.0) and p["p_wilcoxon"] < 1e-3
    same = paired(b, b)
    assert same["ties"] == 20 and same["p_wilcoxon"] == 1.0


def test_paired_ignores_unshared_instances_and_table_runs():
    rs = rows("A", [5, 6, 7, 8]) + rows("B", [6, 6, 8, 9, 100])
    out = table(rs, ref="B")
    assert "A" in out and "B" in out and "W/T/L" in out
    a = {r["inst"]: r for r in rs if r["cfg"] == "A"}
    b = {r["inst"]: r for r in rs if r["cfg"] == "B"}
    assert paired(a, b)["n"] == 4


@needs_l2br
def test_l2br_sampler_never_beats_the_optimum():
    from brp.l2br_adapter import load_model
    from brp.l2br_sampler import l2br_best_of_m
    model = load_model(ROOT / "IL/final_models/CVS-uBRP/H=5xS=5.pt", "u", "cpu", ROOT)
    rng = random.Random(4)
    gaps = []
    for _ in range(4):
        inst = random_instance(rng, 5, 7, 10)
        opt, _ = solve(inst, 7, "u")
        obj = l2br_best_of_m(model, inst, 7, "u", M=64)
        assert obj is not None and obj >= opt     # a sampled solution can match but never beat the optimum
        gaps.append(obj - opt)
    assert np.mean(gaps) <= 2                     # and with M=64 it should be close
