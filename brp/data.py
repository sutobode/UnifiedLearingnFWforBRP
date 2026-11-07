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


# ---------------------------------------------------------------------------------------------------
# ZQLZ (Zhu, Qin, Lim, Zhang 2012): header "S N", then per stack "<stack index> p1 ... pk" (bottom -> top).
# Folders "<T>-<S>-<N>" hold 100 instances each; T is the tier limit.
# ---------------------------------------------------------------------------------------------------
def parse_zqlz_file(path) -> tuple:
    lines = Path(path).read_text().splitlines()
    S, N = map(int, lines[0].split())
    stacks = tuple(tuple(int(x) for x in ln.split()[1:]) for ln in lines[1:1 + S])
    assert len(stacks) == S and sum(map(len, stacks)) == N, f"{path}: header/stack mismatch"
    return stacks


def load_zqlz(T: int, S: int, root=DEFAULT_L2BR) -> list:
    """[(key, stacks)] for all folders T-S-*, ordered by folder then file."""
    base = Path(root) / "datasets" / "ZQLZ"
    out = []
    for folder in sorted(f for f in base.iterdir() if f.is_dir() and f.name.startswith(f"{T}-{S}-")):
        for f in sorted(folder.glob("*.txt")):
            out.append((f"{folder.name}/{f.stem}", parse_zqlz_file(f)))
    if not out:
        raise FileNotFoundError(f"no ZQLZ instances for T={T}, S={S} under {base}")
    return out


# ---------------------------------------------------------------------------------------------------
# LL (Lee & Lee 2010): header "name bays rows tiers N N"; per stack "bay stack ntiers c1 c1 c2 c2 ..."
# (every container is listed twice). Several bays are merged into S = bays*rows stacks.
# Files "<R|U><bays:02><rows:02><tiers:02>_<N>_<k>.txt"; R = random, U = upside down.
# ---------------------------------------------------------------------------------------------------
def parse_ll_file(path) -> tuple:
    lines = Path(path).read_text().splitlines()
    head = lines[0].split()
    bays, rows, tiers, N = int(head[1]), int(head[2]), int(head[3]), int(head[4])
    stacks = [()] * (bays * rows)
    for ln in lines[1:]:
        v = [int(x) for x in ln.split()]
        if len(v) < 3:
            continue
        stacks[(v[0] - 1) * rows + (v[1] - 1)] = tuple(dict.fromkeys(v[3:]))
    stacks = tuple(stacks)
    assert sum(map(len, stacks)) == N, f"{path}: container count mismatch"
    return stacks, tiers


def load_ll(kind: str, T: int, S: int, root=DEFAULT_L2BR, rows: int = 16) -> list:
    """kind 'R' or 'U'. S = bays*16. Returns [(key, stacks)]."""
    base = Path(root) / "datasets" / "LL" / ("Individual, random" if kind == "R" else "Individual, upside down")
    pat = re.compile(rf"{kind}{S // rows:02d}{rows:02d}{T:02d}_")
    out = []
    for f in sorted(base.glob("*.txt")):
        if pat.match(f.name):
            stacks, tiers = parse_ll_file(f)
            assert tiers == T
            out.append((f.stem, stacks))
    if not out:
        raise FileNotFoundError(f"no LL-{kind} instances for T={T}, S={S} under {base}")
    return out


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
