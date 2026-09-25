"""
Trajectory stores.

A *store* is the only thing in this project that knows how bytes are laid out
on disk.  Everything above it (samplers, datasets, tasks) sees exactly one
interface:

    store.n_traj                      -> int
    store.T, store.H, store.W         -> int
    store.channel_names               -> list[str]
    store.dt                          -> float   (seconds per frame)
    store.read(traj, t_idx, ch_idx)   -> np.ndarray (len(t_idx), H, W, len(ch_idx))

`read` takes an explicit list of frame indices, NOT a slice.  That is the whole
point: a direct-time sample needs frames [t0-K+1 .. t0] plus the single frame
[t0+h].  Reading them as one strided gather costs K+1 frames regardless of how
large h is, which is what makes horizon-balanced sampling affordable at h=512.

Two backends are provided:

    ZarrStore    RealPDEBench combustion, converted by the user's existing
                 `convert_combustion_to_zarr.py`:
                     <root>/data           (N, T, H, W, C) float32
                     <root>/channels.json  {"channel_names": [...]}
    NPYStore     one .npy per trajectory, shape (T, H, W, C).  Used for REALM
                 IgnitHIT / EvolveJet and for the synthetic smoke-test data.

Both are fork- and spawn-safe: the on-disk handle is opened lazily inside each
DataLoader worker and never pickled.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class TrajectoryStore:
    """Read-only random access to (N, T, H, W, C) trajectory data."""

    n_traj: int
    T: int
    H: int
    W: int
    channel_names: List[str]
    dt: float

    @property
    def n_channels(self) -> int:
        return len(self.channel_names)

    def read(self, traj: int, t_idx: Sequence[int],
             ch_idx: Optional[Sequence[int]] = None) -> np.ndarray:
        raise NotImplementedError

    def read_traj_channel(self, traj: int, ch: int,
                          t_stride: int = 1, t_max: Optional[int] = None) -> np.ndarray:
        """(T', H, W) for one channel — used by the data audit and by POD."""
        t_max = self.T if t_max is None else min(t_max, self.T)
        t_idx = list(range(0, t_max, t_stride))
        return self.read(traj, t_idx, [ch])[..., 0]

    def summary(self) -> str:
        return (f"{type(self).__name__}: N={self.n_traj} T={self.T} "
                f"H={self.H} W={self.W} C={self.n_channels} dt={self.dt:g}s")


# ---------------------------------------------------------------------------
# Zarr
# ---------------------------------------------------------------------------

class ZarrStore(TrajectoryStore):
    """RealPDEBench combustion, zarr backend."""

    def __init__(self, path: str, dt: float = 2.5e-4,
                 channel_names: Optional[Sequence[str]] = None):
        self.path = path
        self.dt = float(dt)
        self._handle = None                       # opened lazily per process

        arr = self._arr()
        n, T, H, W, C = arr.shape
        self.n_traj, self.T, self.H, self.W = int(n), int(T), int(H), int(W)
        self.channel_names = (list(channel_names) if channel_names is not None
                              else self._load_channel_names(path, self._root(), C))
        if len(self.channel_names) != C:
            raise ValueError(f"{len(self.channel_names)} channel names for {C} "
                             f"channels in {path}")
        # Drop the handle again so the object is picklable before workers fork.
        self._handle = None

    # -- lazy handle ------------------------------------------------------
    def _root(self):
        if self._handle is None:
            import zarr
            self._handle = zarr.open(self.path, mode="r")
        return self._handle

    def _arr(self):
        return self._root()["data"]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    @staticmethod
    def _load_channel_names(path: str, root, n_channels: int) -> List[str]:
        sidecar = os.path.join(path, "channels.json")
        if os.path.exists(sidecar):
            with open(sidecar) as fp:
                return list(json.load(fp)["channel_names"])
        attrs = dict(root.attrs)
        if "channel_names" in attrs:
            return list(attrs["channel_names"])
        raise FileNotFoundError(
            f"No channel names for {path}: expected {sidecar} or a "
            f"`channel_names` attribute on the zarr group.")

    # -- read -------------------------------------------------------------
    def read(self, traj, t_idx, ch_idx=None) -> np.ndarray:
        arr = self._arr()
        t_idx = np.asarray(t_idx, dtype=np.int64)
        ch_idx = (np.arange(self.n_channels) if ch_idx is None
                  else np.asarray(ch_idx, dtype=np.int64))

        # zarr 2.x accepts fancy indexing on a single axis through __getitem__;
        # zarr 3.x wants `.oindex` for orthogonal (outer) selections.  Frames
        # are usually contiguous-ish, so we gather the enclosing slab once and
        # index it in numpy — one store round-trip instead of len(t_idx).
        t_lo, t_hi = int(t_idx.min()), int(t_idx.max()) + 1
        span = t_hi - t_lo
        gather_slab = span <= max(16, 4 * len(t_idx))

        if gather_slab:
            oidx = getattr(arr, "oindex", None)
            if oidx is not None:
                block = oidx[traj, slice(t_lo, t_hi), :, :, ch_idx]
            else:                                   # zarr 2.x
                block = arr[traj, t_lo:t_hi, :, :, ch_idx]
            block = np.asarray(block)
            out = block[t_idx - t_lo]
        else:
            oidx = getattr(arr, "oindex", None)
            if oidx is not None:
                out = np.asarray(oidx[traj, t_idx, :, :, ch_idx])
            else:
                out = np.stack([np.asarray(arr[traj, int(t), :, :, ch_idx])
                                for t in t_idx], axis=0)
        return np.ascontiguousarray(out, dtype=np.float32)


# ---------------------------------------------------------------------------
# One .npy per trajectory
# ---------------------------------------------------------------------------

class NPYStore(TrajectoryStore):
    """Directory of per-trajectory `.npy` files, each (T, H, W, C).

    Used for REALM IgnitHIT / EvolveJet after conversion, and by
    `scripts/make_synthetic_data.py` for the end-to-end smoke test.  Files are
    memory-mapped, so a horizon-512 gather still touches only K+1 frames.
    """

    def __init__(self, path: str, dt: float = 1.0,
                 channel_names: Optional[Sequence[str]] = None,
                 pattern: str = "*.npy"):
        import glob
        self.path = path
        self.dt = float(dt)
        self.files = sorted(glob.glob(os.path.join(path, pattern)))
        if not self.files:
            raise FileNotFoundError(f"no files matching {pattern} under {path}")
        self._mmaps: dict = {}

        head = np.load(self.files[0], mmap_mode="r")
        T, H, W, C = head.shape
        self.n_traj, self.T, self.H, self.W = len(self.files), int(T), int(H), int(W)

        if channel_names is not None:
            self.channel_names = list(channel_names)
        else:
            sidecar = os.path.join(path, "channels.json")
            if os.path.exists(sidecar):
                with open(sidecar) as fp:
                    meta = json.load(fp)
                self.channel_names = list(meta["channel_names"])
                self.dt = float(meta.get("dt", self.dt))
            else:
                self.channel_names = [f"ch{i}" for i in range(C)]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_mmaps"] = {}
        return state

    def _mm(self, traj: int):
        mm = self._mmaps.get(traj)
        if mm is None:
            mm = np.load(self.files[traj], mmap_mode="r")
            self._mmaps[traj] = mm
        return mm

    def read(self, traj, t_idx, ch_idx=None) -> np.ndarray:
        mm = self._mm(traj)
        t_idx = np.asarray(t_idx, dtype=np.int64)
        out = np.asarray(mm[t_idx])                       # (t, H, W, C)
        if ch_idx is not None:
            out = out[..., np.asarray(ch_idx, dtype=np.int64)]
        return np.ascontiguousarray(out, dtype=np.float32)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class StridedStore(TrajectoryStore):
    """Temporal subsampling: frame i of this store is frame i*stride of `base`.

    The point is a controlled experiment the two public datasets cannot give on
    their own. RealPDEBench samples at 4 kHz and Lifted H2 at 200 kHz, and the
    AR-vs-direct crossover sits at h ~ 3 on the first and h ~ 22 on the second.
    Two datasets, two sampling rates, two different physics -- nothing in that
    comparison isolates which of the three moved the crossover.

    Striding RealPDEBench by 1, 2, 4, 8 holds the physics and the trajectories
    fixed and changes ONLY dt. If the crossover measured in physical time stays
    put across strides, it is a property of the flow; if it stays put in FRAMES,
    it is a property of the model and the K-frame history. That is a
    single-variable experiment, and it costs four training runs rather than a
    new dataset.

    dt is scaled with the stride, so tau stays physical and every horizon axis
    remains comparable across strides.
    """

    def __init__(self, base: TrajectoryStore, stride: int = 1):
        self.base = base
        self.stride = int(stride)
        if self.stride < 1:
            raise ValueError("time_stride must be >= 1")
        self.n_traj = base.n_traj
        self.T = base.T // self.stride
        self.H, self.W = base.H, base.W
        self.channel_names = list(base.channel_names)
        self.dt = base.dt * self.stride

    def read(self, traj, t_idx, ch_idx=None):
        phys = [int(t) * self.stride for t in t_idx]
        return self.base.read(traj, phys, ch_idx)

    def summary(self) -> str:
        return (f"StridedStore(stride={self.stride}) over " + self.base.summary()
                + f"  ->  T={self.T} dt={self.dt:g}s")


def build_store(cfg: dict) -> TrajectoryStore:
    """cfg keys: backend ('zarr'|'npy'), data_path, dt, time_stride, channel_names?"""
    backend = str(cfg.get("backend", "zarr")).lower()
    path = cfg["data_path"]
    #dt = float(cfg.get("dt", 2.5e-4))
    _dt = cfg.get("dt", 2.5e-4)
    dt = float(_dt) if _dt is not None else None
    names = cfg.get("all_channel_names", None)
    if backend == "zarr":
        store = ZarrStore(path, dt=dt, channel_names=names)
    elif backend == "well":
        from .well import WellStore
        store = WellStore.from_config(cfg)
    elif backend in ("arrow", "hf_arrow"):
        from .arrow_store import ArrowTrajectoryStore
        store = ArrowTrajectoryStore.from_config(cfg)
    elif backend in ("npy", "numpy"):
        store = NPYStore(path, dt=dt, channel_names=names,
                         pattern=cfg.get("file_pattern", "*.npy"))
    else:
        raise ValueError(f"unknown backend '{backend}' (expected zarr | npy)")

    stride = int(cfg.get("time_stride", 1))
    return store if stride == 1 else StridedStore(store, stride)
