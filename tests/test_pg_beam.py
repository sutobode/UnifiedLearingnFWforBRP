import math
import random
from pathlib import Path

import pytest

from brp.data import DEFAULT_L2BR, random_instance
from brp.exact import solve
from brp.search.pg_beam import pg_beam
from brp.sim import actions, replay, settle

T, S = 4, 3


def uniform_policy(variant):
    def fn(states):
        out = []
        for st in states:
            acts = actions(st, T, variant)
            out.append([(a, -math.log(len(acts))) for a in acts])
        return out
    return fn


@pytest.mark.parametrize("variant", ["u", "r"])
@pytest.mark.parametrize("seed", range(15))
def test_wide_beam_matches_exact_optimum(variant, seed):
    rng = random.Random(100 + seed)
    inst = random_instance(rng, S, T, rng.randint(4, 8))
    opt, _ = solve(inst, T, variant)
    res = pg_beam(uniform_policy(variant), inst, T, variant, B=100000, k=100, lam=0.0)
    assert res.obj == opt                                  # exhaustive level-wise search is optimal
    assert replay(inst, T, variant, res.path) == res.obj    # and the path is a valid solution


def test_upper_bound_pruning_semantics():
    rng = random.Random(5)
    inst = random_instance(rng, S, T, 8)
    opt, _ = solve(inst, T, "u")
    policy = uniform_policy("u")
    assert pg_beam(policy, inst, T, "u", B=100000, k=100, ub=opt).obj is None       # nothing strictly better
    assert pg_beam(policy, inst, T, "u", B=100000, k=100, ub=opt + 1).obj == opt    # finds the optimum


def test_narrow_beam_returns_valid_solution_and_terminates():
    rng = random.Random(11)
    for variant in ("u", "r"):
        inst = random_instance(rng, 5, 7, 25)
        res = pg_beam(uniform_policy_for(5, 7, variant), inst, 7, variant, B=8, k=2, lam=1.0)
        assert res.obj is not None and replay(inst, 7, variant, res.path) == res.obj


def uniform_policy_for(S_, T_, variant):
    def fn(states):
        return [[(a, 0.0) for a in actions(st, T_, variant)] for st in states]
    return fn


needs_model = pytest.mark.skipif(not (Path(DEFAULT_L2BR) / "IL" / "final_models").exists(),
                                 reason="third_party/L2BR missing")


@needs_model
def test_with_real_l2br_policy_valid_solution():
    from brp.l2br_adapter import load_model, log_probs
    root = Path(DEFAULT_L2BR)
    model = load_model(root / "IL/final_models/CVS-uBRP/H=5xS=5.pt", "u", "cpu", root)
    inst = random_instance(random.Random(2), 5, 7, 25)
    res = pg_beam(lambda sts: log_probs(model, sts, 7, "u"), inst, 7, "u", B=16, k=3, lam=1.0)
    assert res.obj is not None and replay(inst, 7, "u", res.path) == res.obj
