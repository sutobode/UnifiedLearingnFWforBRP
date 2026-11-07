import random
from pathlib import Path

import pytest

from brp.bounds import lb1
from brp.data import DEFAULT_L2BR, cvs_tiers, load_cvs, random_instance
from brp.exact import solve
from brp.sim import actions, canonical, is_done, replay, settle, step

# Fig. 1 of the L2BR paper: 3 stacks, 4 tiers, 9 blocks (bottom -> top)
FIG1 = ((1, 3, 4), (8, 6, 9), (7, 5, 2))
FIG1_SOLUTION = [(0, 2), (0, 1), (2, 0), (1, 0)]  # the 4 relocations described in the paper


def test_fig1_paper_solution_replays_with_four_relocations():
    assert replay(FIG1, 4, "r", FIG1_SOLUTION) == 4
    assert replay(FIG1, 4, "u", FIG1_SOLUTION) == 4


def test_fig1_exact_not_worse_than_paper_example():
    opt_r, path_r = solve(FIG1, 4, "r")
    opt_u, path_u = solve(FIG1, 4, "u")
    assert opt_u <= opt_r <= 4
    assert replay(FIG1, 4, "r", path_r) == opt_r
    assert replay(FIG1, 4, "u", path_u) == opt_u


def test_settle_retrieves_chain():
    assert is_done(settle(((2, 1), (4, 3))))
    assert settle(((1, 2), (3,))) == ((1, 2), (3,))  # 1 is under 2: nothing retrievable


def test_actions_respect_capacity_and_variant():
    st = settle(((3, 1, 2), (5, 4), ()))  # T=3: stack 0 full
    acts_u = actions(st, 3, "u")
    assert all(d != s and len(st[d]) < 3 for s, d in acts_u)
    assert (0, 1) in acts_u and (1, 2) in acts_u
    acts_r = actions(st, 3, "r")
    assert {s for s, _ in acts_r} == {0}  # only the target stack's top block moves


def test_invalid_replay_raises():
    with pytest.raises(ValueError):
        replay(FIG1, 4, "r", [(1, 0)])  # not the target stack in rBRP
    with pytest.raises(ValueError):
        replay(FIG1, 4, "r", FIG1_SOLUTION[:2])  # does not finish


def test_lb1_known_values():
    assert lb1(FIG1) == 3  # 3 and 4 above 1 in stack 0; 9 above 6 in stack 1
    assert lb1(((1, 2, 3),)) == 2  # 2 and 3 sit above smaller 1
    assert lb1(((3, 2, 1),)) == 0  # increasing urgency upwards: nothing blocks
    assert lb1(((),)) == 0


@pytest.mark.parametrize("seed", range(40))
def test_random_small_invariants(seed):
    rng = random.Random(seed)
    S, T = 3, 4
    N = rng.randint(3, 8)
    inst = random_instance(rng, S, T, N)
    opt_u, path_u = solve(inst, T, "u")
    opt_r, path_r = solve(inst, T, "r")
    assert lb1(settle(inst)) <= opt_u <= opt_r  # admissible bound; uBRP is a relaxation of rBRP
    assert replay(inst, T, "u", path_u) == opt_u
    assert replay(inst, T, "r", path_r) == opt_r
    perm = list(inst)
    rng.shuffle(perm)  # stack order must not change the optimum
    assert solve(tuple(perm), T, "u")[0] == opt_u
    assert canonical(tuple(perm)) == canonical(inst)


def test_random_instance_feasibility_bound():
    rng = random.Random(0)
    with pytest.raises(AssertionError):
        random_instance(rng, 3, 4, 4 * 3)  # completely full bay


needs_data = pytest.mark.skipif(not (Path(DEFAULT_L2BR) / "datasets" / "CVS").exists(),
                                reason="run scripts/setup_l2br.py first (third_party/L2BR missing)")


@needs_data
def test_cvs_loading_counts():
    insts = load_cvs(3, 3)
    assert len(insts) == 40 and all(sum(map(len, i)) == 9 for i in insts)


@needs_data
def test_cvs_h3_s3_matches_paper_tables():
    """Paper Table 2 (uBRP, Algo-TS exact) = 4.975 and Table 5 (rBRP, IP exact) = 5.000 for CVS H=3, S=3."""
    insts, T = load_cvs(3, 3), cvs_tiers(3)
    mean_u = sum(solve(i, T, "u")[0] for i in insts) / len(insts)
    mean_r = sum(solve(i, T, "r")[0] for i in insts) / len(insts)
    assert mean_u == pytest.approx(4.975, abs=1e-9)
    assert mean_r == pytest.approx(5.000, abs=1e-9)
