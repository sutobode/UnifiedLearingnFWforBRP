"""Exact solver for small BRP instances: A* with the admissible bound lb1 and symmetry merging."""
from __future__ import annotations

import heapq
from itertools import count

from .bounds import lb1
from .sim import actions, canonical, is_done, settle, step


class NodeLimit(Exception):
    pass


def solve(stacks, T: int, variant: str, max_nodes: int | None = None):
    """Return (optimal #relocations, action list). lb1 is admissible but not necessarily consistent, so a
    state is re-expanded whenever a cheaper path to it is found."""
    start = settle(tuple(tuple(s) for s in stacks))
    if is_done(start):
        return 0, []
    tie = count()
    nodes = [(start, -1, None)]  # (state, parent index, action) for path reconstruction
    best_g = {canonical(start): 0}
    heap = [(lb1(start), 0, next(tie), 0)]  # (f, g, tie, node index)
    expanded = 0
    while heap:
        f, g, _, idx = heapq.heappop(heap)
        state = nodes[idx][0]
        if g > best_g.get(canonical(state), g):
            continue
        if is_done(state):
            path, j = [], idx
            while nodes[j][1] != -1:
                path.append(nodes[j][2])
                j = nodes[j][1]
            return g, path[::-1]
        expanded += 1
        if max_nodes is not None and expanded > max_nodes:
            raise NodeLimit(f"exceeded {max_nodes} expansions")
        for a in actions(state, T, variant):
            nxt = step(state, a)
            key = canonical(nxt)
            ng = g + 1
            if ng < best_g.get(key, 1 << 30):
                best_g[key] = ng
                nodes.append((nxt, idx, a))
                heapq.heappush(heap, (ng + lb1(nxt), ng, next(tie), len(nodes) - 1))
    raise RuntimeError("no solution found (instance infeasible?)")
