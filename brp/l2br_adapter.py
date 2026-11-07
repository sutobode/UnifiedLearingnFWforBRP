"""Adapter to the public L2BR policy (third_party/L2BR, pinned commit) for use from our own simulator.

* patch_l2br(): make the original code device-agnostic (it hard-codes 'cuda:0' in encoder.py/decoder.py).
  Idempotent; touches only the (git-ignored) clone, never our repo.
* encode(): our state (tuple of stacks of priorities) -> the model's input tensor (B, S, T) with values
  (n+1-r)/(n+1), r = rank of the block among the n remaining blocks (1 = retrieved first), 0 = empty slot.
  This equals what L2BR's Env.clear() produces after its running renormalisation (checked in tests).
* log_probs(): log-probabilities over OUR valid (src, dst) actions (variant 'u') or destination stacks
  expressed as (target_stack, dst) actions (variant 'r').
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from .sim import actions as valid_actions
from .sim import target_stack

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "L2BR"
_REPLACEMENTS = {
    "encoder.py": [(".to('cuda:0')", ".to(x.device)")],
    "decoder.py": [("return newnode_embeddings.to(device)", "return newnode_embeddings.to(node_embeddings.device)"),
                   ("return torch.cat([extd_grpah_embedding, node_embedding], dim=2).to(device)",
                    "return torch.cat([extd_grpah_embedding, node_embedding], dim=2).to(node_embedding.device)")],
}


def patch_l2br(root=DEFAULT_ROOT) -> bool:
    """Returns True if files were modified, False if already patched."""
    root, changed = Path(root), False
    for fname, reps in _REPLACEMENTS.items():
        p = root / fname
        src = p.read_text(encoding="utf-8")
        for old, new in reps:
            if old in src:
                src = src.replace(old, new)
                changed = True
            elif new not in src:
                raise RuntimeError(f"{fname}: expected pattern not found: {old!r} (L2BR version changed?)")
        p.write_text(src, encoding="utf-8")
    return changed


ACTIVE_ROOT = DEFAULT_ROOT  # the L2BR clone in use; set by load_model so other modules follow it (Kaggle path!)


def load_model(model_path, variant: str, device="cpu", root=DEFAULT_ROOT, embed_dim=128, n_heads=8,
               n_layers=3, ff=512):
    global ACTIVE_ROOT
    ACTIVE_ROOT = Path(root)
    patch_l2br(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from decoder import L2BR  # noqa: E402  (module of the cloned repo)
    args = SimpleNamespace(problem_type="uBRP" if variant == "u" else "rBRP", embed_dim=embed_dim,
                           n_heads=n_heads, n_encode_layers=n_layers, ff_hidden_dim=ff)
    model = L2BR(args, device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    return model.to(device).eval()


def encode(states, T: int, device="cpu") -> torch.Tensor:
    S = len(states[0])
    x = torch.zeros(len(states), S, T)
    for b, st in enumerate(states):
        flat = sorted(p for s in st for p in s)
        n = len(flat)
        rank = {p: i + 1 for i, p in enumerate(flat)}
        for i, s in enumerate(st):
            for t, p in enumerate(s):
                x[b, i, t] = (n + 1 - rank[p]) / (n + 1)
    return x.to(device)


@torch.no_grad()
def log_probs(model, states, T: int, variant: str, device="cpu", temperature: float = 1.0) -> list:
    """For each (settled, non-empty) state: list of ((src, dst), logprob) over its valid actions."""
    S = len(states[0])
    logits = model(encode(states, T, device)).float().cpu()  # (B, S*S) for 'u'; (B, S) for 'r'
    out = []
    for b, st in enumerate(states):
        acts = valid_actions(st, T, variant)
        if variant == "u":
            idx = [s * S + d for s, d in acts]
        else:
            idx = [d for _, d in acts]
        lp = torch.log_softmax(logits[b, idx] / temperature, dim=0)
        out.append(list(zip(acts, lp.tolist())))
    return out
