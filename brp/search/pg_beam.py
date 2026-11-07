"""PG-Beam: policy-guided, bound-aware beam search for the BRP.

Depth-synchronous beam: every relocation costs 1, so all nodes of a level have the same g. At each level the
beam (width B) is expanded with the top-k actions of a learned policy; children are
  * dropped if the cleared bay is already seen at the same or smaller depth (canonical key -> no cycles, so the
    search terminates),
  * dropped if g + lb1 >= UB (they cannot improve the incumbent upper bound; lb1 is admissible),
  * ranked by  lb1(child) - lam * cumulative_logprob   (lower is better), keeping the best B.
The first terminal child found has the minimum depth reachable inside the beam, so it is returned at once.
lam = 0 gives a pure bound-driven beam (BBS-like); a large lam follows the policy.

policy_fn(states) -> for every state a list [((src, dst), logprob), ...] over its valid actions
(see brp.l2br_adapter.log_probs). Keeping it a plain callable makes the search testable without a network.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..bounds import lb1
from ..sim import canonical, is_done, settle, step


@dataclass
class BeamResult:
    obj: int | None  # relocations of the solution found, or None if nothing better than ub was found
    path: list | None
    levels: int = 0
    expanded: int = 0
    nn_calls: int = 0
    wall_s: float = 0.0
    timed_out: bool = False
    stats: dict = field(default_factory=dict)


def _beam_once(policy_fn, inst, T: int, variant: str, B: int, k: int, lam: float,
               ub: int | None, time_limit_s: float | None, max_levels: int) -> BeamResult:
    t0 = time.time()
    start = settle(tuple(tuple(s) for s in inst))
    if is_done(start):
        return BeamResult(0, [], wall_s=time.time() - t0)
    best_ub = float("inf") if ub is None else ub
    nodes = [(start, -1, None)]            # (state, parent index, action)
    beam = [(0, 0.0)]                      # (node index, cumulative logprob)
    seen = {canonical(start): 0}
    res = BeamResult(None, None)
    for level in range(max_levels):
        if not beam:
            break
        if time_limit_s is not None and time.time() - t0 > time_limit_s:
            res.timed_out = True
            break
        states = [nodes[i][0] for i, _ in beam]
        lps = policy_fn(states)
        res.nn_calls += 1
        g = level + 1
        cands = []
        for (idx, clp), acts in zip(beam, lps):
            for a, lp in sorted(acts, key=lambda x: -x[1])[:k]:
                res.expanded += 1
                nxt = step(nodes[idx][0], a)
                if is_done(nxt):
                    if g < best_ub:
                        nodes.append((nxt, idx, a))
                        path, j = [], len(nodes) - 1
                        while nodes[j][1] != -1:
                            path.append(nodes[j][2])
                            j = nodes[j][1]
                        res.obj, res.path, res.levels = g, path[::-1], g
                        res.wall_s = time.time() - t0
                        return res
                    continue
                h = lb1(nxt)
                if g + h >= best_ub:
                    continue
                key = canonical(nxt)
                if seen.get(key, 1 << 30) <= g:
                    continue
                seen[key] = g
                nodes.append((nxt, idx, a))
                cands.append((h - lam * (clp + lp), len(nodes) - 1, clp + lp))
        cands.sort(key=lambda c: c[0])
        beam = [(i, clp) for _, i, clp in cands[:B]]
        res.levels = g
    res.wall_s = time.time() - t0
    return res


def pg_beam(policy_fn, inst, T: int, variant: str, B: int = 64, k: int = 3, lam: float = 1.0,
            ub: int | None = None, time_limit_s: float | None = None, max_levels: int = 2000,
            widen: bool = True, max_widen: int = 8) -> BeamResult:
    """Beam search with iterative widening. Without an incumbent (ub is None) a narrow beam can run dry
    (all children already seen); then B and k are doubled and the search restarts, so a solution is always
    returned (the limit is an exhaustive level-wise search). With an incumbent ub, returning obj=None means
    'no solution better than ub inside this beam' and the caller keeps its incumbent."""
    t0 = time.time()
    total = BeamResult(None, None)
    for attempt in range(max_widen + 1):
        left = None if time_limit_s is None else max(0.0, time_limit_s - (time.time() - t0))
        r = _beam_once(policy_fn, inst, T, variant, B, k, lam, ub, left, max_levels)
        total.expanded += r.expanded
        total.nn_calls += r.nn_calls
        if r.obj is not None or ub is not None or not widen or r.timed_out:
            r.expanded, r.nn_calls, r.wall_s = total.expanded, total.nn_calls, time.time() - t0
            r.stats = {"attempts": attempt + 1, "B": B, "k": k}
            return r
        B, k = B * 2, k * 2
    total.wall_s = time.time() - t0
    total.stats = {"attempts": max_widen + 1, "B": B, "k": k}
    return total
