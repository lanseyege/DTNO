"""
Semigroup consistency (§31).

    C_SG(t_a, t_b) = || Phi(z0, t_a + t_b) - Phi(Phi(z0, t_a), t_b) ||
                     / || Phi(z0, t_a + t_b) ||

Measured for DT-FNO and SG-DT-FNO on the same anchors and the same (t_a, t_b)
grid.  If the semigroup training does anything, C_SG for Model C should be far
below Model B.

That, on its own, proves nothing.  §31 is explicit about the trap: a latent that
has partially collapsed is consistent for free.  So this module reports three
things together and `scripts/make_figures.py` plots them together:

    C_SG            the consistency itself
    E_test          the forecasting error on the same anchors
    latent_rms      the scale of z, as a collapse tripwire

The claim we are entitled to make is only the joint one — C_SG down AND E_test
down.  C_SG down with E_test flat means the constraint bought consistency in the
latent and nothing in the physics, which is Negative Result B in §44 and a
perfectly publishable outcome, just not the one we are hoping for.

Also here: `field_consistency`, the decoded version.  Latent agreement is what
is trained; field agreement is what a reader cares about, and the two can differ
if the decoder is contractive.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch


DEFAULT_TAU_PAIRS = [(1, 1), (2, 2), (4, 4), (8, 8), (16, 16), (32, 32),
                     (4, 28), (28, 4), (16, 48), (48, 16)]


@torch.no_grad()
def semigroup_consistency(model, x: torch.Tensor, h_a: int, h_b: int,
                          dt: float, t_scale: float,
                          grid: Optional[torch.Tensor] = None,
                          eps: float = 1e-8) -> Dict[str, float]:
    """One (h_a, h_b) pair on one batch.  Returns latent and field consistency."""
    B = x.shape[0]
    dev = x.device
    tau_a = torch.full((B,), h_a * dt / t_scale, device=dev)
    tau_b = torch.full((B,), h_b * dt / t_scale, device=dev)

    z0 = model.encode(x, grid)
    z_direct = model.propagate(z0, tau_a + tau_b)
    z_compose = model.propagate(model.propagate(z0, tau_a), tau_b)

    def rel(a, b):
        af, bf = a.float().reshape(B, -1), b.float().reshape(B, -1)
        return float((torch.linalg.vector_norm(af - bf, dim=1)
                      / (torch.linalg.vector_norm(af, dim=1) + eps)).mean())

    u_direct = model.decode(z_direct, x)
    u_compose = model.decode(z_compose, x)

    return {
        "h_a": h_a, "h_b": h_b, "h_total": h_a + h_b,
        "C_SG": rel(z_direct, z_compose),
        "C_SG_field": rel(u_direct, u_compose),
        "latent_rms": float(z_direct.float().pow(2).mean().sqrt()),
        "latent_var_batch": float(
            z_direct.float().reshape(B, -1).var(dim=0, unbiased=False).mean()),
    }


@torch.no_grad()
def evaluate_semigroup(model, loader, dt: float, t_scale: float,
                       tau_pairs: Optional[Sequence[Tuple[int, int]]] = None,
                       device="cuda", max_batches: Optional[int] = None,
                       distributed: bool = False) -> Dict[str, object]:
    """Sweep (h_a, h_b) over a loader; returns per-pair means."""
    pairs = list(tau_pairs or DEFAULT_TAU_PAIRS)
    sums: Dict[Tuple[int, int], Dict[str, float]] = {p: {} for p in pairs}
    counts: Dict[Tuple[int, int], int] = {p: 0 for p in pairs}

    model.eval()
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = batch["x"].to(device, non_blocking=True)
        grid = batch.get("grid")
        grid = grid.to(device) if grid is not None else None
        for p in pairs:
            rec = semigroup_consistency(model, x, p[0], p[1], dt, t_scale, grid)
            for k, v in rec.items():
                sums[p][k] = sums[p].get(k, 0.0) + float(v)
            counts[p] += 1

    if distributed:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            keys = sorted({k for p in pairs for k in sums[p]})
            dev = torch.device(device)
            buf = torch.tensor(
                [[sums[p].get(k, 0.0) for k in keys] + [float(counts[p])]
                 for p in pairs], dtype=torch.float64, device=dev)
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            rows = buf.tolist()
            for i, p in enumerate(pairs):
                counts[p] = int(rows[i][-1])
                sums[p] = {k: rows[i][j] for j, k in enumerate(keys)}

    out: List[Dict[str, float]] = []
    for p in pairs:
        n = max(counts[p], 1)
        rec = {k: v / n for k, v in sums[p].items()}
        rec["h_a"], rec["h_b"], rec["h_total"] = p[0], p[1], p[0] + p[1]
        out.append(rec)
    return {
        "pairs": out,
        "C_SG_mean": sum(r["C_SG"] for r in out) / max(len(out), 1),
        "C_SG_field_mean": sum(r["C_SG_field"] for r in out) / max(len(out), 1),
        "latent_rms_mean": sum(r["latent_rms"] for r in out) / max(len(out), 1),
    }


@torch.no_grad()
def identity_consistency(model, loader, device="cuda",
                         max_batches: int = 8) -> Dict[str, float]:
    """|| Phi(z0, 0) - z0 || / || z0 || — the §17 constraint, measured."""
    tot, n = 0.0, 0
    model.eval()
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x = batch["x"].to(device, non_blocking=True)
        grid = batch.get("grid")
        grid = grid.to(device) if grid is not None else None
        z0 = model.encode(x, grid)
        z_id = model.identity_latent(z0)
        B = z0.shape[0]
        a = z0.float().reshape(B, -1)
        b = z_id.float().reshape(B, -1)
        tot += float((torch.linalg.vector_norm(a - b, dim=1)
                      / (torch.linalg.vector_norm(a, dim=1) + 1e-8)).mean())
        n += 1
    return {"C_identity": tot / max(n, 1)}
