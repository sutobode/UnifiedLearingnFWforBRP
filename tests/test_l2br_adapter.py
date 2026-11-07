"""Checks that our simulator + encoding agree with L2BR's own Env, and that its action masks match ours.
Skipped when third_party/L2BR is missing (run it once: git clone + checkout pinned commit, see CLAUDE.md)."""
import random
import sys

import pytest
import torch

from brp.data import DEFAULT_L2BR, random_instance
from brp.l2br_adapter import encode, load_model, log_probs, patch_l2br
from brp.sim import actions, is_done, settle, step

ROOT = DEFAULT_L2BR
pytestmark = pytest.mark.skipif(not (ROOT / "env.py").exists(), reason="third_party/L2BR missing")
S, T, N = 5, 7, 25  # CVS H=5, S=5


def env_cls():
    patch_l2br(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from env import Env
    return Env


def raw_x(inst):
    n = sum(map(len, inst))
    x = torch.zeros(1, len(inst), T)
    for i, s in enumerate(inst):
        for t, p in enumerate(s):
            x[0, i, t] = ((n + 1) - p) / (n + 1)  # exactly what L2BR's file loader produces
    return x


@pytest.mark.parametrize("variant,path", [("u", "IL/final_models/CVS-uBRP/H=5xS=5.pt"),
                                          ("r", "IL/final_models/CVS-rBRP/H=5xS=5.pt")])
def test_encoding_masks_and_transitions_match_l2br(variant, path):
    Env = env_cls()
    model = load_model(ROOT / path, variant, "cpu", ROOT)
    rng = random.Random(7)
    for _ in range(5):
        inst = random_instance(rng, S, T, N)
        env = Env("cpu", raw_x(inst))
        env.clear()
        mine = settle(inst)
        for _step in range(60):
            if is_done(mine):
                assert env.empty.all()
                break
            x_mine = encode([mine], T)
            assert torch.allclose(x_mine, env.x, atol=1e-5), "state encoding differs from L2BR Env"
            logits = model(x_mine)[0]
            valid = set(torch.nonzero(logits > -1e8).flatten().tolist())
            acts = actions(mine, T, variant)
            expected = {s * S + d for s, d in acts} if variant == "u" else {d for _, d in acts}
            assert valid == expected, "L2BR action mask differs from our valid actions"
            s, d = rng.choice(acts)
            if variant == "u":
                env.step_uBRP(torch.tensor([[s, d]]))
            else:
                env.step_rBRP(torch.tensor([[d]]))
            mine = step(mine, (s, d))
        assert is_done(mine) or _step == 59


def test_log_probs_are_a_distribution_over_valid_actions():
    model = load_model(ROOT / "IL/final_models/CVS-uBRP/H=5xS=5.pt", "u", "cpu", ROOT)
    st = settle(random_instance(random.Random(3), S, T, N))
    (lp,) = log_probs(model, [st], T, "u")
    assert {a for a, _ in lp} == set(actions(st, T, "u"))
    assert abs(sum(torch.tensor([x for _, x in lp]).exp()) - 1) < 1e-4
