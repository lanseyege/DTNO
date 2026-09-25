"""
Train / validation / test splits (§9).

The leak this file exists to prevent: windows (U_100:104 -> U_120) and
(U_105:109 -> U_121) come from the same trajectory and are almost the same
sample.  Splitting windows at random puts near-duplicates on both sides of the
wall and every model looks excellent.

Primary protocol is therefore **trajectory-held-out**: 20 train / 5 val / 5 test
of RealPDEBench's 30 numerical trajectories, with a test trajectory appearing in
training under no time window whatsoever.  The split is computed once, written
to JSON, and loaded from that JSON forever after — including by the audit that
computes normalisation statistics, which must see training trajectories only.

`stratified_split` groups trajectories by whatever scalar metadata is available
(fuel composition, equivalence ratio) and deals round-robin within each group so
the three sets cover the same parameter range.  With no metadata it falls back to
a seeded shuffle, which is the honest thing to do rather than pretending to
stratify.

The secondary within-trajectory protocol (§9) is a temporal cut on the same
trajectories, reported for comparability with published benchmark numbers but
never used to select a model.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class Split:
    train: List[int]
    val: List[int]
    test: List[int]
    protocol: str = "trajectory_held_out"
    seed: int = 42
    note: str = ""
    # within-trajectory protocol only: fraction of frames reserved at the tail
    time_cut: Optional[Dict[str, List[int]]] = None

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fp:
            json.dump(asdict(self), fp, indent=2)

    @staticmethod
    def load(path: str) -> "Split":
        with open(path) as fp:
            return Split(**json.load(fp))

    def check_disjoint(self):
        s = [set(self.train), set(self.val), set(self.test)]
        for i in range(3):
            for j in range(i + 1, 3):
                overlap = s[i] & s[j]
                if overlap:
                    raise ValueError(f"split leak: trajectories {sorted(overlap)} "
                                     f"appear in two subsets")

    def summary(self) -> str:
        return (f"[{self.protocol}] train={self.train} val={self.val} "
                f"test={self.test}")


def stratified_split(n_traj: int,
                     n_val: int = 5,
                     n_test: int = 5,
                     seed: int = 42,
                     strata: Optional[Sequence] = None,
                     note: str = "") -> Split:
    """Deterministic split, stratified by `strata[i]` when it is provided."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n_traj)

    if strata is not None and len(strata) == n_traj:
        buckets: Dict[str, List[int]] = {}
        for i, s in enumerate(strata):
            buckets.setdefault(str(s), []).append(int(i))
        order: List[int] = []
        keys = sorted(buckets)
        for k in keys:                       # shuffle inside a stratum only
            b = np.array(buckets[k])
            rng.shuffle(b)
            buckets[k] = b.tolist()
        # Round-robin across strata so consecutive picks come from different
        # parameter groups; slicing that order then balances all three subsets.
        while any(buckets[k] for k in keys):
            for k in keys:
                if buckets[k]:
                    order.append(buckets[k].pop())
        note = note or f"stratified over {len(keys)} metadata groups"
    else:
        order = idx.tolist()
        rng.shuffle(order)
        note = note or "no metadata found; seeded random assignment"

    test = sorted(order[:n_test])
    val = sorted(order[n_test:n_test + n_val])
    train = sorted(order[n_test + n_val:])
    sp = Split(train=train, val=val, test=test, seed=seed, note=note)
    sp.check_disjoint()
    return sp


def within_trajectory_split(n_traj: int, T: int, train_frac: float = 0.7,
                            val_frac: float = 0.15) -> Split:
    """Secondary protocol: same trajectories, disjoint time ranges (§9)."""
    t_train = int(T * train_frac)
    t_val = int(T * (train_frac + val_frac))
    allt = list(range(n_traj))
    return Split(train=allt, val=allt, test=allt,
                 protocol="within_trajectory",
                 note="temporal cut; report only, never used for selection",
                 time_cut={"train": [0, t_train],
                           "val": [t_train, t_val],
                           "test": [t_val, T]})


def load_or_create_split(path: str, n_traj: int, cfg: Optional[dict] = None,
                         strata: Optional[Sequence] = None) -> Split:
    """Load a frozen split, or create + freeze one on first call."""
    cfg = cfg or {}
    if os.path.exists(path):
        sp = Split.load(path)
        sp.check_disjoint()
        return sp
    sp = stratified_split(n_traj,
                          n_val=int(cfg.get("n_val_traj", 5)),
                          n_test=int(cfg.get("n_test_traj", 5)),
                          seed=int(cfg.get("split_seed", 42)),
                          strata=strata)
    sp.save(path)
    return sp
