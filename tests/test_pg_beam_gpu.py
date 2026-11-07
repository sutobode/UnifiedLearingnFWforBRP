import random
from pathlib import Path

import pytest
import torch

from brp.bounds import lb1
from brp.data import DEFAULT_L2BR, random_instance
from brp.exact import solve
from brp.search.pg_beam_gpu import (apply_move_t, encode_t, lb1_t, make_hash_params, pg_beam_gpu, settle_t,
                                    state_keys, valid_mask_t)
from brp.sim import actions, canonical, is_done, replay, settle, step

S, T = 4, 5


def to_tensor(states):
    st = torch.zeros(len(states), S, T, dtype=torch.int32)
    for b, state in enumerate(states):
        for i, s in enumerate(state):
            for t, p in enumerate(s):
                st[b, i, t] = p
    return st


def to_stacks(row):
    return tuple(tuple(int(v) for v in s if v > 0) for s in row)


def random_states(n, seed, with_settle):
    """Raw priorities (not rank-compressed): states reached by random play from random instances."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        st = random_instance(rng, S, T, rng.randint(6, 14))
        st = settle(st)
        for _ in range(rng.randint(0, 6)):
            if is_done(st):
                break
            st = step(st, rng.choice(actions(st, T, rng.choice("ur"))))
        if not is_done(st):
            out.append(st if with_settle else tuple(tuple(s) for s in random_instance(rng, S, T, 8)))
    return out


def test_settle_and_lb1_match_reference():
    raw = [tuple(tuple(s) for s in random_instance(random.Random(i), S, T, random.Random(i).randint(5, 15)))
           for i in range(60)]
    st = settle_t(to_tensor(raw))
    for b, inst in enumerate(raw):
        assert to_stacks(st[b]) == settle(inst)
    states = random_states(80, 1, True)
    lb = lb1_t(to_tensor(states))
    assert [int(x) for x in lb] == [lb1(s) for s in states]


@pytest.mark.parametrize("variant", ["u", "r"])
def test_valid_mask_and_apply_move_match_reference(variant):
    states = random_states(60, 2, True)
    st = to_tensor(states)
    mask = valid_mask_t(st, variant)
    for b, state in enumerate(states):
        acts = actions(state, T, variant)
        idx = {s * S + d for s, d in acts} if variant == "u" else {d for _, d in acts}
        assert {int(i) for i in torch.nonzero(mask[b]).flatten()} == idx
        s, d = random.Random(b).choice(acts)
        ch = apply_move_t(st, torch.tensor([b]), torch.tensor([s]), torch.tensor([d]))
        assert to_stacks(ch[0]) == step(state, (s, d))


def test_encode_matches_adapter_formula():
    # rank-compressed instance: v = (N+1-p)/(n+1) equals the rank-based encoding of l2br_adapter.encode
    from brp.l2br_adapter import encode
    rng = random.Random(3)
    inst = random_instance(rng, S, T, 12)
    st = settle(inst)
    for _ in range(4):
        if is_done(st):
            break
        st = step(st, rng.choice(actions(st, T, "u")))
    if not is_done(st):
        assert torch.allclose(encode_t(to_tensor([st]), 12), encode([st], T), atol=1e-6)


def test_state_keys_canonical():
    R, Q = make_hash_params(S, T, "cpu")
    states = random_states(300, 4, True)
    keys = state_keys(to_tensor(states), R, Q)
    perm = [tuple(random.Random(i).sample(list(s), len(s))) for i, s in enumerate(states)]
    assert torch.equal(keys, state_keys(to_tensor(perm), R, Q))        # stack order does not matter
    by_canon = {}
    for s, k in zip(states, keys.tolist()):
        by_canon.setdefault(canonical(s), set()).add(k)
    assert all(len(v) == 1 for v in by_canon.values())                 # same state -> same key
    assert len({next(iter(v)) for v in by_canon.values()}) == len(by_canon)   # no collisions


def uniform(variant, Sn):
    A = Sn * Sn if variant == "u" else Sn
    return lambda x: torch.zeros(x.shape[0], A)


@pytest.mark.parametrize("variant", ["u", "r"])
@pytest.mark.parametrize("seed", range(12))
def test_wide_gpu_beam_matches_exact_optimum(variant, seed):
    rng = random.Random(200 + seed)
    Sn, Tn = 3, 4
    inst = random_instance(rng, Sn, Tn, rng.randint(4, 8))
    opt, _ = solve(inst, Tn, variant)
    res = pg_beam_gpu(uniform(variant, Sn), inst, Tn, variant, B=100000, k=100, lam=0.0)
    assert res.obj == opt and replay(inst, Tn, variant, res.path) == opt


def test_ub_semantics_and_narrow_beam_valid():
    rng = random.Random(9)
    inst = random_instance(rng, 3, 4, 8)
    opt, _ = solve(inst, 4, "u")
    assert pg_beam_gpu(uniform("u", 3), inst, 4, "u", B=100000, k=100, ub=opt).obj is None
    assert pg_beam_gpu(uniform("u", 3), inst, 4, "u", B=100000, k=100, ub=opt + 1).obj == opt
    for variant in ("u", "r"):
        big = random_instance(rng, 5, 7, 25)
        r = pg_beam_gpu(uniform(variant, 5), big, 7, variant, B=8, k=2, lam=1.0)
        assert r.obj is not None and replay(big, 7, variant, r.path) == r.obj


@pytest.mark.skipif(not (Path(DEFAULT_L2BR) / "IL" / "final_models").exists(), reason="third_party/L2BR missing")
def test_real_policy_gpu_beam_valid_and_comparable_to_python_beam():
    from brp.l2br_adapter import load_model, log_probs
    from brp.search.pg_beam import pg_beam
    root = Path(DEFAULT_L2BR)
    model = load_model(root / "IL/final_models/CVS-uBRP/H=5xS=5.pt", "u", "cpu", root)
    fn = lambda x: model(x)
    for seed in range(3):
        inst = random_instance(random.Random(seed), 5, 7, 25)
        g = pg_beam_gpu(fn, inst, 7, "u", B=16, k=3, lam=1.0)
        p = pg_beam(lambda sts: log_probs(model, sts, 7, "u"), inst, 7, "u", B=16, k=3, lam=1.0)
        assert replay(inst, 7, "u", g.path) == g.obj
        assert abs(g.obj - p.obj) <= 3      # same algorithm; tie-breaking/dedup order may differ slightly
