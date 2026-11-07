"""Admissible lower bound for the BRP (valid for both rBRP and uBRP)."""
from __future__ import annotations


def lb1(stacks) -> int:
    """Number of blocks lying above a block with a smaller priority in the same stack.

    Each such block must be relocated at least once before the block below it can be retrieved, and one
    relocation moves one block, so the remaining optimum is >= lb1. Linear in the number of blocks.
    """
    total = 0
    for s in stacks:
        lowest = None
        for p in s:  # bottom -> top
            if lowest is not None and p > lowest:
                total += 1
            lowest = p if lowest is None else min(lowest, p)
    return total
