"""Reference simulator for the Block Relocation Problem (pure Python, immutable states).

State = tuple of stacks; a stack = tuple of priorities, bottom -> top. Smaller priority = retrieved earlier.
Priorities are distinct (no duplicates). A relocation moves the top block of one stack onto another stack
with free space; cost = 1 per relocation. Retrieval is automatic: whenever the smallest remaining priority
is on top of its stack it is removed (free).

variant 'u' (unrestricted): any top block may move to any other non-full stack.
variant 'r' (restricted)  : only the top block of the stack holding the smallest priority (the blocking block).
"""
from __future__ import annotations

Stacks = tuple  # tuple[tuple[int, ...], ...]
Action = tuple  # (src, dst)


def settle(stacks: Stacks) -> Stacks:
    """Retrieve every block that is currently retrievable (free)."""
    cur = [list(s) for s in stacks]
    while True:
        nonempty = [s for s in cur if s]
        if not nonempty:
            break
        m = min(min(s) for s in nonempty)
        for s in cur:
            if s and s[-1] == m:
                s.pop()
                break
        else:
            break
    return tuple(tuple(s) for s in cur)


def is_done(stacks: Stacks) -> bool:
    return not any(stacks)


def n_blocks(stacks: Stacks) -> int:
    return sum(len(s) for s in stacks)


def target_stack(stacks: Stacks) -> int:
    """Index of the stack holding the smallest remaining priority."""
    best, idx = None, -1
    for i, s in enumerate(stacks):
        if s:
            m = min(s)
            if best is None or m < best:
                best, idx = m, i
    return idx


def actions(stacks: Stacks, T: int, variant: str) -> list:
    """Valid (src, dst) relocations. `stacks` must already be settled and non-empty."""
    S = len(stacks)
    if variant == "r":
        src = target_stack(stacks)
        return [(src, d) for d in range(S) if d != src and len(stacks[d]) < T]
    return [(s, d) for s in range(S) if stacks[s]
            for d in range(S) if d != s and len(stacks[d]) < T]


def step(stacks: Stacks, action: Action) -> Stacks:
    """Apply one relocation, then retrieve automatically."""
    s, d = action
    cur = [list(x) for x in stacks]
    cur[d].append(cur[s].pop())
    return settle(tuple(tuple(x) for x in cur))


def canonical(stacks: Stacks) -> Stacks:
    """Stacks are interchangeable for the objective, so sort them to merge symmetric states."""
    return tuple(sorted(stacks))


def replay(stacks: Stacks, T: int, variant: str, acts: list) -> int:
    """Replay an action sequence from an instance; validate every move; return #relocations.
    Raises ValueError for an invalid move or if the sequence does not clear the bay."""
    cur = settle(tuple(tuple(s) for s in stacks))
    for k, a in enumerate(acts):
        if is_done(cur):
            raise ValueError(f"extra action #{k} after the bay was already empty")
        if tuple(a) not in set(actions(cur, T, variant)):
            raise ValueError(f"invalid action #{k}: {a}")
        cur = step(cur, tuple(a))
    if not is_done(cur):
        raise ValueError("action sequence ends before all blocks are retrieved")
    return len(acts)
