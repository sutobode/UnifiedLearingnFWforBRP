"""Vectorised (torch) PG-Beam: the same search as pg_beam.py, with simulation, bounds, hashing and
de-duplication done as tensor operations on the beam, so only ~tens of kernel launches per level remain.

State tensor st: int32 (C, S, T), 0 = empty slot, bottom at tier 0, tiers contiguous. Priorities are
rank-compressed to 1..N at the start, so after removing the k smallest, the model input of the original
L2BR code is v = (N + 1 - p) / (n + 1) with n the number of blocks left (checked against brp.l2br_adapter.encode).

logits_fn(x) -> (C, A) raw logits for x = encode_t(...): A = S*S (uBRP, index src*S + dst) or A = S (rBRP,
destination stack). Invalid actions are re-masked here, so a plain callable (even a uniform one) is enough.
"""
from __future__ import annotations

import time

import torch

from .pg_beam import BeamResult

BIG = 1 << 20


def settle_t(st: torch.Tensor) -> torch.Tensor:
    """Automatic retrieval, vectorised: repeatedly remove the globally smallest priority while it is on top."""
    st = st.clone()
    while True:
        pos = st > 0
        L = pos.sum(2)
        vals = torch.where(pos, st, torch.full_like(st, BIG))
        minp = vals.amin(dim=(1, 2))
        top = torch.gather(st, 2, (L - 1).clamp(min=0).unsqueeze(2)).squeeze(2)
        top = torch.where(L > 0, top, torch.full_like(top, BIG + 1))
        hit = (top == minp[:, None]) & (minp[:, None] < BIG)
        if not bool(hit.any()):
            return st
        c, s = hit.nonzero(as_tuple=True)
        st[c, s, L[c, s] - 1] = 0


def lb1_t(st: torch.Tensor) -> torch.Tensor:
    """Blocks lying above a smaller priority in the same stack (admissible lower bound), per state."""
    pos = st > 0
    vals = torch.where(pos, st, torch.full_like(st, BIG))
    runmin = torch.cummin(vals, dim=2).values
    prev = torch.cat([torch.full_like(runmin[:, :, :1], BIG), runmin[:, :, :-1]], dim=2)
    return (pos & (st > prev)).sum(dim=(1, 2))


def target_stack_t(st: torch.Tensor) -> torch.Tensor:
    vals = torch.where(st > 0, st, torch.full_like(st, BIG))
    return vals.amin(dim=2).argmin(dim=1)


def valid_mask_t(st: torch.Tensor, variant: str) -> torch.Tensor:
    C, S, T = st.shape
    L = (st > 0).sum(2)
    dst_ok = L < T
    if variant == "u":
        m = (L > 0)[:, :, None] & dst_ok[:, None, :] & ~torch.eye(S, dtype=torch.bool, device=st.device)[None]
        return m.reshape(C, S * S)
    src = target_stack_t(st)
    return dst_ok & (torch.arange(S, device=st.device)[None, :] != src[:, None])


def encode_t(st: torch.Tensor, N: int) -> torch.Tensor:
    n = (st > 0).sum(dim=(1, 2)).float()
    return torch.where(st > 0, (N + 1 - st).float() / (n + 1)[:, None, None], torch.zeros((), device=st.device))


def apply_move_t(st: torch.Tensor, parent: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    ch = st[parent].clone()
    K = ch.shape[0]
    ar = torch.arange(K, device=st.device)
    L = (ch > 0).sum(2)
    val = ch[ar, src, L[ar, src] - 1]
    ch[ar, src, L[ar, src] - 1] = 0
    ch[ar, dst, L[ar, dst]] = val
    return settle_t(ch)


def make_hash_params(S: int, T: int, device, seed: int = 12345):
    g = torch.Generator().manual_seed(seed)
    R = torch.randint(-2 ** 62, 2 ** 62, (T,), generator=g, dtype=torch.int64) | 1
    Q = torch.randint(-2 ** 62, 2 ** 62, (S,), generator=g, dtype=torch.int64) | 1
    return R.to(device), Q.to(device)


def state_keys(st: torch.Tensor, R: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """64-bit key of the canonical state (stacks sorted), equal for stack-permuted states."""
    h = (st.long() * R[None, None, :]).sum(2)
    hs = torch.sort(h, dim=1).values
    return (hs * Q[None, :]).sum(1)


def _to_tensor(inst, T: int, device):
    flat = sorted(p for s in inst for p in s)
    rank = {p: i + 1 for i, p in enumerate(flat)}
    st = torch.zeros(1, len(inst), T, dtype=torch.int32, device=device)
    for i, s in enumerate(inst):
        for t, p in enumerate(s):
            st[0, i, t] = rank[p]
    return st, len(flat)


@torch.no_grad()
def _beam_once(logits_fn, inst, T, variant, B, k, lam, ub, time_limit_s, device, max_levels) -> BeamResult:
    t0 = time.time()
    st0, N = _to_tensor(inst, T, device)
    st0 = settle_t(st0)
    if int((st0 > 0).sum()) == 0:
        return BeamResult(0, [], wall_s=time.time() - t0)
    S = st0.shape[1]
    best_ub = float("inf") if ub is None else ub
    R, Q = make_hash_params(S, T, device)
    beam, clp = st0, torch.zeros(1, device=device)
    seen = state_keys(st0, R, Q)
    hist, res = [], BeamResult(None, None)
    for level in range(max_levels):
        C = beam.shape[0]
        if C == 0:
            break
        if time_limit_s is not None and time.time() - t0 > time_limit_s:
            res.timed_out = True
            break
        mask = valid_mask_t(beam, variant)
        logits = logits_fn(encode_t(beam, N)).float().masked_fill(~mask, -1e9)
        lp = torch.log_softmax(logits, dim=1)
        top_lp, top_a = lp.topk(min(k, lp.shape[1]), dim=1)
        ok = mask.gather(1, top_a)
        res.nn_calls += 1
        parent = torch.arange(C, device=device)[:, None].expand_as(top_a)[ok]
        a = top_a[ok]
        new_clp = (clp[:, None] + top_lp)[ok]
        if variant == "u":
            src, dst = a // S, a % S
        else:
            src, dst = target_stack_t(beam)[parent], a
        child = apply_move_t(beam, parent, src, dst)
        res.expanded += int(child.shape[0])
        g = level + 1
        term = (child > 0).sum(dim=(1, 2)) == 0
        if bool(term.any()) and g < best_ub:
            j = int(torch.where(term, new_clp, torch.full_like(new_clp, -1e30)).argmax())
            path = [(int(src[j]), int(dst[j]))]
            idx = int(parent[j])
            for lvl in range(level - 1, -1, -1):
                par, s_, d_ = hist[lvl]
                path.append((int(s_[idx]), int(d_[idx])))
                idx = int(par[idx])
            res.obj, res.path, res.levels = g, path[::-1], g
            res.wall_s = time.time() - t0
            return res
        h = lb1_t(child)
        keys = state_keys(child, R, Q)
        keep = (~term) & (g + h < best_ub) & ~torch.isin(keys, seen)
        if not bool(keep.any()):
            beam = child[:0]
            break
        child, parent, src, dst = child[keep], parent[keep], src[keep], dst[keep]
        new_clp, h, keys = new_clp[keep], h[keep], keys[keep]
        order = torch.argsort(h.float() - lam * new_clp)
        uniq, inv = torch.unique(keys[order], return_inverse=True)
        first = torch.full((uniq.shape[0],), order.shape[0], dtype=torch.long, device=device)
        first = first.scatter_reduce(0, inv, torch.arange(order.shape[0], device=device), "amin", include_self=True)
        sel = order[first.sort().values[:B]]
        beam, clp = child[sel], new_clp[sel]
        hist.append((parent[sel], src[sel], dst[sel]))
        seen = torch.cat([seen, keys[sel]])
        res.levels = g
    res.wall_s = time.time() - t0
    return res


def pg_beam_gpu(logits_fn, inst, T: int, variant: str, B: int = 256, k: int = 3, lam: float = 1.0,
                ub: int | None = None, time_limit_s: float | None = None, device="cpu",
                max_levels: int = 2000, widen: bool = True, max_widen: int = 8) -> BeamResult:
    """Same contract as brp.search.pg_beam.pg_beam (iterative widening when there is no incumbent)."""
    t0 = time.time()
    total = BeamResult(None, None)
    for attempt in range(max_widen + 1):
        left = None if time_limit_s is None else max(0.0, time_limit_s - (time.time() - t0))
        r = _beam_once(logits_fn, inst, T, variant, B, k, lam, ub, left, device, max_levels)
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
