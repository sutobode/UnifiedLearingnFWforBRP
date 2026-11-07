"""One entry point for the three public benchmarks and the matching public L2BR models."""
from __future__ import annotations

from pathlib import Path

from .data import DEFAULT_L2BR, cvs_tiers, load_cvs, load_ll, load_zqlz


def load_benchmark(name: str, a: int, S: int, root=DEFAULT_L2BR):
    """name: 'CVS' (a = H, T = H+2), 'ZQLZ' (a = T), 'LL-R' / 'LL-U' (a = T).
    Returns (items [(key, stacks)], T)."""
    if name == "CVS":
        return [(f"CVS/H{a}S{S}/inst{i + 1:02d}", st) for i, st in enumerate(load_cvs(a, S, root))], cvs_tiers(a)
    if name == "ZQLZ":
        return load_zqlz(a, S, root), a
    if name in ("LL-R", "LL-U"):
        return load_ll(name[-1], a, S, root), a
    raise ValueError(name)


def l2br_model_path(name: str, variant: str, a: int, S: int, root=DEFAULT_L2BR) -> Path:
    """Public SIL model of the paper for this setting (variant 'u' or 'r')."""
    root, v = Path(root), ("uBRP" if variant == "u" else "rBRP")
    if name == "CVS":
        return root / "SIL" / "final_models" / f"CVS-{v}" / f"H={a}xS={S}.pt"
    if name == "ZQLZ":
        return root / "SIL" / "final_models" / f"ZQLZ-{v}" / f"T={a}xS={S}.pt"
    return root / "SIL" / "final_models" / f"LL-{v}" / f"({name[-1]})T={a}xS={S}.pt"
