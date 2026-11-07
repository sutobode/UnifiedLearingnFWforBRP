"""Instance loading (public benchmarks) and random instance generation.

CVS files (Caserta, Schwarze, Voss 2012): first line "S N"; then one line per stack: "count p1 ... pk"
(priorities bottom -> top, smaller = retrieved earlier). Tier limit for CVS is T = H + 2.
"""
from __future__ import annotations

import random
import re
from pathlib import Path

DEFAULT_L2BR = Path(__file__).resolve().parents[1] / "third_party" / "L2BR"


def parse_cvs_file(path) -> tuple:
    lines = Path(path).read_text().splitlines()
    S, N = map(int, lines[0].split())
    stacks = tuple(tuple(int(x) for x in lines[i].split()[1:]) for i in range(1, S + 1))
    assert sum(len(s) for s in stacks) == N, f"{path}: block count mismatch"
    return stacks


def load_cvs(H: int, S: int, root=DEFAULT_L2BR) -> list:
    """The 40 CVS instances with H tiers' worth of blocks per stack and S stacks, ordered by file index."""
    d = Path(root) / "datasets" / "CVS"
    files = sorted((f for f in d.iterdir() if re.fullmatch(rf"data{H}-{S}-\d+\.dat", f.name)),
                   key=lambda f: int(f.stem.rsplit("-", 1)[1]))
    if not files:
        raise FileNotFoundError(f"no CVS files for H={H}, S={S} under {d}")
    return [parse_cvs_file(f) for f in files]


def cvs_tiers(H: int) -> int:
    return H + 2


def random_instance(rng: random.Random, S: int, T: int, N: int) -> tuple:
    """Random bay with N distinct priorities 1..N in S stacks of capacity T (needs N <= T*S - T + 1 so that
    a relocation is always possible)."""
    assert N <= T * S - T + 1, "bay too full to guarantee feasibility"
    prio = list(range(1, N + 1))
    rng.shuffle(prio)
    stacks = [[] for _ in range(S)]
    for p in prio:
        stacks[rng.choice([i for i in range(S) if len(stacks[i]) < T])].append(p)
    return tuple(tuple(s) for s in stacks)
