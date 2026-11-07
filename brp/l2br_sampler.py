"""L2BR's own best-of-M sampling, per instance (the paper's ESS: M lock-step rollouts of the policy on the
original Env, stop when the first rollout empties the bay). Gives per-instance baseline objectives and times
for paired statistics. Mirrors third_party/L2BR/ess.py::validation (which only reports the mean)."""
from __future__ import annotations

import sys

import torch

from .l2br_adapter import DEFAULT_ROOT, patch_l2br


def _env_cls(root=None):
    from . import l2br_adapter
    root = root or l2br_adapter.ACTIVE_ROOT      # follows the clone passed to load_model()
    patch_l2br(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from env import Env
    return Env


def raw_x(inst, T: int) -> torch.Tensor:
    """Input exactly as L2BR's file loaders build it (before Env.clear())."""
    n = sum(map(len, inst))
    x = torch.zeros(1, len(inst), T)
    for i, s in enumerate(inst):
        for t, p in enumerate(s):
            x[0, i, t] = ((n + 1) - p) / (n + 1)
    return x


@torch.no_grad()
def l2br_best_of_m(model, inst, T: int, variant: str, M: int, temp: float = 1.0, device="cpu",
                   traj_limit: int = 500, root=None):
    """Number of relocations of the first rollout to finish (= best of M), or None if none finished within
    traj_limit. Rollouts run in lock-step, so the first to finish has the fewest relocations."""
    Env = _env_cls(root)
    S = len(inst)
    env = Env(device, raw_x(inst, T).to(device).repeat(M, 1, 1))
    env.clear()
    for step in range(traj_limit):
        if env.empty.any().item():
            return step
        p = torch.softmax(model(env.x) / temp, dim=1)
        a = torch.multinomial(p, 1)
        if variant == "r":
            env.step_rBRP(a)
        else:
            env.step_uBRP(torch.cat((a // S, a % S), 1))
    return None
