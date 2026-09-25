"""
The Well (Polymathic AI) — HDF5 backend for the DTNO trajectory-store contract.

This is the shared machinery behind `data/gray_scott.py` and
`data/rayleigh_benard.py`.  It is deliberately a *store*, not a Dataset: once
it satisfies

    store.n_traj / T / H / W / channel_names / dt
    store.read(traj, t_idx, ch_idx) -> (len(t_idx), H, W, len(ch_idx))

every existing piece of the pipeline — horizon sampler, DirectPairDataset,
EvalAnchorDataset, audit, climatology baselines, spectra — works unchanged.
That is the whole point of §41: the architecture, losses, protocol and
evaluation must not change between datasets, so a new dataset may only add a
store and a defaults module.


ONE DTNO TRAJECTORY = ONE (file, realization) PAIR
--------------------------------------------------
The Well lays out each file as

    /t0_fields/<name>     (n_traj, T, *spatial)
    /t1_fields/<name>     (n_traj, T, *spatial, D)
    /t2_fields/<name>     (n_traj, T, *spatial, D, D)     <- refused, see below
    /scalars/<name>       per-file physical parameters
    /dimensions/time      (T,)

so a file is an operating point holding many *independent* realizations.  The
loader this module replaces concatenated those realizations along the time axis
and carried `traj_bounds` so windowing would not straddle a boundary.  Here we
do not need that trick and deliberately do not use it: a DTNO trajectory is one
(file, realization) pair, full stop.  A concatenated time axis would let
`DirectPairDataset` draw an anchor near the end of realization j and a target
inside realization j+1 — a discontinuity no operator can predict and no error
model can explain — and the guard against it would be a second, parallel notion
of "trajectory" that `EvalAnchorDataset` and the climatology baselines know
nothing about.  Flattening instead is free: `n_traj` becomes
n_files x n_realizations and the split, which is per-trajectory anyway, becomes
strictly finer-grained.

Vector fields are flattened into channels (`velocity` -> `velocity_x`,
`velocity_y`) so the flat channel list the rest of the pipeline expects comes
out of discovery rather than out of a hard-coded table.  Rank-2 tensor fields
are refused rather than flattened by guesswork; nothing in this study needs one.


WHAT YOU MUST CHECK ON A NEW WELL DATASET
-----------------------------------------
1. The grouping `scripts/audit_data.py` prints.  `data/channels.py` matches
   channel names with loose regexes, and The Well spells physics differently
   again ("buoyancy", "A", "B").  A channel that lands in the fallback group is
   still reported, but the §26 per-variable table stops being informative.
   `channel_rename` in the config is the intended fix — see the configs.
2. The spatial slicing line this store prints.  `spatial_stride` subsamples an
   axis; `spatial_crop` is the OUTPUT shape.  Rayleigh-Benard is 512 x 128 and
   wants (128, 128), which is stride (4, 1) — a *scalar* stride of 4 would ask
   for 512 raw points on an axis that only has 128 and now raises instead of
   quietly returning a 128 x 32 field.
3. Whether the grid is uniform.  The Well's `rayleigh_benard` is sampled at
   Chebyshev nodes along z; `rayleigh_benard_uniform` is the resampled copy.
   An FNO's FFT and the §27 spectral metric both assume uniform spacing, so on
   the Chebyshev grid the vertical spectrum is not a physical spectrum.  This
   store warns when it sees a non-uniform `dimensions` axis.


PROCESSES, HANDLES AND HDF5
---------------------------
`DataLoader(num_workers > 0)` forks on Linux, and forking does not call
`__getstate__`, so a worker inherits whatever the parent had open.  An
inherited HDF5 handle is unusable -- libhdf5 keeps process-global state that
fork duplicates rather than shares -- so the handle cache here is keyed on
`(pid, file index)` and a forked child always opens its own.  See `_h`.

Two things that are the caller's job, not this module's:

  * **File locking.**  On NFS, Lustre, or any filesystem where HDF5's advisory
    locks misbehave, dozens of workers opening the same file read-only can
    block or fail with "unable to lock file".  Export
    `HDF5_USE_FILE_LOCKING=FALSE` before `torchrun` if you see that; it must be
    set before h5py is imported, so setting it from Python is usually too late.
  * **Thread pinning.**  `data/realpde._worker_init` pins each worker to one
    thread, but the `OMP_NUM_THREADS` half of it runs after fork, by which
    point libgomp has already cached the value it read at library load in the
    parent.  Export the thread variables in the shell alongside `torchrun`;
    `run/21_expA_train.sh` does.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .store import TrajectoryStore

VECTOR_SUFFIX = ("x", "y", "z")
SPLIT_DIRS = ("train", "valid", "val", "test")


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def discover_fields(path: str, fields: Optional[Sequence[str]] = None
                    ) -> Tuple[List[str], List[tuple], Tuple[int, ...], int, int]:
    """Walk one file -> (channel_names, spec, spatial, n_traj, T).

    `spec` entries are (group, dataset, component | None) in channel order, so
    `read` never has to re-derive which HDF5 dataset a channel came from.
    """
    import h5py

    names: List[str] = []
    spec: List[tuple] = []
    spatial: Optional[Tuple[int, ...]] = None
    n_traj = T = None

    with h5py.File(path, "r") as f:
        n_spatial = int(f.attrs.get("n_spatial_dims", 0)) or None
        for grp, order in (("t0_fields", 0), ("t1_fields", 1)):
            if grp not in f:
                continue
            for key in sorted(f[grp]):
                if fields is not None and key not in fields:
                    continue
                shp = tuple(int(s) for s in f[grp][key].shape)
                if spatial is None:
                    ns = n_spatial if n_spatial is not None else (
                        len(shp) - 2 if order == 0 else len(shp) - 3)
                    spatial = shp[2:2 + ns]
                    n_traj, T = shp[0], shp[1]
                ncomp = 1 if order == 0 else int(shp[-1])
                if ncomp == 1:
                    names.append(key)
                    spec.append((grp, key, None))
                else:
                    for c in range(ncomp):
                        suf = VECTOR_SUFFIX[c] if c < 3 else str(c)
                        names.append(f"{key}_{suf}")
                        spec.append((grp, key, c))
        if "t2_fields" in f:
            keep = [k for k in f["t2_fields"]
                    if fields is None or k in fields]
            if keep:
                raise NotImplementedError(
                    f"{os.path.basename(path)} has rank-2 fields {keep}. "
                    f"Flattening them needs a convention no dataset in this "
                    f"study requires; name the fields you want explicitly with "
                    f"`data.fields` instead of guessing.")
    if spatial is None:
        raise RuntimeError(f"no t0/t1 fields found in {path}"
                           + (f" matching {list(fields)}" if fields else ""))
    if len(spatial) != 2:
        raise NotImplementedError(
            f"{os.path.basename(path)} is {len(spatial)}D with spatial shape "
            f"{spatial}. Every model in this project is 2D; slice it to 2D in "
            f"a converter and record which plane you took, rather than having "
            f"the store choose one silently.")
    return names, spec, spatial, int(n_traj), int(T)


def _find_files(data_path: str, pattern: str = "*.hdf5"
                ) -> List[Tuple[str, str]]:
    """-> [(path, official_split)], covering both layouts The Well ships.

        <root>/{train,valid,test}/*.hdf5          (rayleigh_benard as downloaded)
        <root>/data/{train,valid,test}/*.hdf5     (gray_scott as downloaded)
        <root>/*.hdf5                             (flat, split unknown)
    """
    out: List[Tuple[str, str]] = []
    for base in (data_path, os.path.join(data_path, "data")):
        for d in SPLIT_DIRS:
            hits = sorted(glob.glob(os.path.join(base, d, pattern)))
            split = "val" if d in ("valid", "val") else d
            out += [(p, split) for p in hits]
    if not out:
        hits = sorted(glob.glob(os.path.join(data_path, pattern)))
        out = [(p, "unknown") for p in hits]
    if not out:
        raise FileNotFoundError(
            f"no files matching {pattern!r} under {data_path} "
            f"(looked in <root>/, <root>/data/, and the "
            f"{{{','.join(SPLIT_DIRS)}}} subdirectories of both)")
    # one path can only be found once even if both bases resolve
    seen, uniq = set(), []
    for p, s in out:
        rp = os.path.realpath(p)
        if rp not in seen:
            seen.add(rp)
            uniq.append((p, s))
    return uniq


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------

class WellStore(TrajectoryStore):
    """Lazy random-frame access to a directory of The Well HDF5 files."""

    def __init__(self,
                 data_path: str,
                 pattern: str = "*.hdf5",
                 fields: Optional[Sequence[str]] = None,
                 channel_rename: Optional[Dict[str, str]] = None,
                 spatial_crop: Optional[Sequence[int]] = None,
                 spatial_stride=1,
                 traj_cut: int = 0,
                 frame_stride: int = 1,
                 max_traj_per_file: Optional[int] = None,
                 max_files: Optional[int] = None,
                 param_scalars: Sequence[str] = (),
                 dt: Optional[float] = None,
                 exclude: Optional[Sequence[Tuple[str, int]]] = None,
                 verbose: bool = True):
        self.data_path = data_path
        self.pattern = pattern
        self.fields = list(fields) if fields else None
        self.traj_cut = int(traj_cut or 0)
        self.frame_stride = max(1, int(frame_stride))
        self._handles: Dict[int, object] = {}

        files = _find_files(data_path, pattern)
        if max_files:
            files = files[:int(max_files)]
        self.files = [p for p, _ in files]
        file_splits = [s for _, s in files]

        names, spec, raw_spatial, n_traj0, T0 = discover_fields(
            self.files[0], self.fields)
        self.spec = spec
        self._raw_spatial = raw_spatial

        # ---- spatial slicing --------------------------------------------
        if np.ndim(spatial_stride) == 0:
            self.spatial_stride = (int(spatial_stride),) * 2
        else:
            self.spatial_stride = tuple(int(v) for v in spatial_stride)
            if len(self.spatial_stride) != 2:
                raise ValueError(f"spatial_stride needs 2 entries for a 2D "
                                 f"grid, got {spatial_stride}")
        self.spatial_crop = (tuple(int(v) for v in spatial_crop)
                             if spatial_crop else None)
        if self.spatial_crop is not None and len(self.spatial_crop) != 2:
            raise ValueError(f"spatial_crop needs 2 entries, got {spatial_crop}")
        self._slices = self._make_slices(raw_spatial)
        out_spatial = tuple(len(range(*sl.indices(n)))
                            for sl, n in zip(self._slices, raw_spatial))
        if self.spatial_crop is not None and out_spatial != self.spatial_crop:
            raise AssertionError(
                f"asked for spatial_crop {self.spatial_crop}; slicing "
                f"{raw_spatial} at stride {self.spatial_stride} gives "
                f"{out_spatial}")
        self.H, self.W = out_spatial

        # ---- channels ----------------------------------------------------
        rename = dict(channel_rename or {})
        self.channel_names = [rename.get(n, n) for n in names]

        # ---- trajectory index --------------------------------------------
        # (file_index, realization_index).  One entry is one DTNO trajectory.
        #
        # Files are keyed "<official_split>/<basename>", not by basename alone.
        # The Well gives the same file name to the same operating point in
        # train/, valid/ and test/, so a bare-basename exclusion list silently
        # drops the named realization from ALL THREE splits -- three times the
        # intended trajectories, with no error and no obvious symptom beyond
        # n_traj being smaller than expected.  Bare basenames are still
        # accepted, but only when they are unambiguous; otherwise this raises
        # rather than guessing which split was meant.
        self.file_keys = [f"{s}/{os.path.basename(p)}"
                          for p, s in zip(self.files, file_splits)]
        drop = self._resolve_exclusions(exclude)
        self.index: List[Tuple[int, int]] = []
        self.traj_split: List[str] = []
        self.traj_file: List[str] = []
        T_per_file: List[int] = []
        n_dropped = 0
        for fi, path in enumerate(self.files):
            n2, _, sp2, nt, T = discover_fields(path, self.fields)
            if [rename.get(n, n) for n in n2] != self.channel_names:
                raise ValueError(
                    f"{os.path.basename(path)} has channels {n2}, but "
                    f"{os.path.basename(self.files[0])} has {names}. Every "
                    f"file in one store must expose the same channels.")
            if sp2 != raw_spatial:
                raise ValueError(f"{os.path.basename(path)} is {sp2}, "
                                 f"expected {raw_spatial}")
            T_per_file.append(T)
            if max_traj_per_file:
                nt = min(nt, int(max_traj_per_file))
            for ti in range(nt):
                if (self.file_keys[fi], ti) in drop:
                    n_dropped += 1
                    continue
                self.index.append((fi, ti))
                self.traj_split.append(file_splits[fi])
                self.traj_file.append(self.file_keys[fi])
        self.n_traj = len(self.index)
        if self.n_traj == 0:
            raise RuntimeError(f"every trajectory under {data_path} was "
                               f"excluded ({n_dropped} dropped)")

        T_raw = min(T_per_file)
        if len(set(T_per_file)) > 1 and verbose:
            print(f"[well] files disagree on T ({sorted(set(T_per_file))}); "
                  f"using the common minimum {T_raw}")
        self.T_raw = T_raw
        self.T = len(range(self.traj_cut, T_raw, self.frame_stride))
        if self.T < 8:
            raise ValueError(
                f"traj_cut={self.traj_cut} frame_stride={self.frame_stride} "
                f"leave T={self.T} usable frames of {T_raw}")

        # ---- physical parameters and dt ----------------------------------
        self.params_by_file = [self._read_scalars(p, param_scalars)
                               for p in self.files]
        self._time_axis = self._read_time(self.files[0])
        self.dt = float(dt) if dt else self._infer_dt()

        if verbose:
            self._report(n_dropped)

    # -- construction from a flat config ----------------------------------
    @classmethod
    def from_config(cls, cfg: dict) -> "WellStore":
        exclude = cfg.get("exclude_trajectories")
        if isinstance(exclude, str):                 # a JSON written by a screen
            with open(exclude) as fp:
                exclude = json.load(fp).get("exclude", [])
        return cls(
            data_path=cfg["data_path"],
            pattern=cfg.get("file_pattern", "*.hdf5"),
            fields=cfg.get("fields"),
            channel_rename=cfg.get("channel_rename"),
            spatial_crop=cfg.get("spatial_crop"),
            spatial_stride=cfg.get("spatial_stride", 1),
            traj_cut=int(cfg.get("traj_cut", 0)),
            frame_stride=int(cfg.get("frame_stride", 1)),
            max_traj_per_file=cfg.get("max_traj_per_file"),
            max_files=cfg.get("max_files"),
            param_scalars=cfg.get("param_scalars", ()),
            dt=cfg.get("dt"),
            exclude=exclude,
            verbose=bool(cfg.get("store_verbose", True)),
        )

    # -- internals ---------------------------------------------------------
    def _resolve_exclusions(self, exclude) -> set:
        """[(file, realization)] -> {(split-qualified key, realization)}.

        `file` may be the split-qualified key ("test/foo.hdf5"), a path, or a
        bare basename.  A bare basename is resolved only when exactly one file
        carries it; when several do, the caller is asked to say which, because
        the alternative is dropping three times as many trajectories as
        intended and finding out from a trajectory count.
        """
        out = set()
        by_base: Dict[str, List[str]] = {}
        for k in self.file_keys:
            by_base.setdefault(k.split("/", 1)[1], []).append(k)
        for raw_name, ti in (exclude or []):
            name = str(raw_name).replace("\\", "/")
            ti = int(ti)
            if name in self.file_keys:
                out.add((name, ti))
                continue
            tail = "/".join(name.split("/")[-2:])
            if tail in self.file_keys:
                out.add((tail, ti))
                continue
            base = name.split("/")[-1]
            hits = by_base.get(base, [])
            if len(hits) == 1:
                out.add((hits[0], ti))
            elif not hits:
                raise KeyError(
                    f"exclusion refers to {raw_name!r}, which is not among "
                    f"the {len(self.file_keys)} files under {self.data_path}")
            else:
                raise KeyError(
                    f"exclusion {raw_name!r} is ambiguous: {len(hits)} files "
                    f"share that name ({hits}). The Well reuses one file name "
                    f"per operating point across train/valid/test, so a bare "
                    f"basename would exclude realization {ti} from every "
                    f"split at once. Qualify it, e.g. '{hits[0]}'. "
                    f"scripts/screen_stationary.py writes qualified keys.")
        return out

    def _make_slices(self, raw):
        out = []
        for i, n in enumerate(raw):
            st = self.spatial_stride[i]
            if self.spatial_crop is None:
                want = (n // st) * st
            else:
                want = self.spatial_crop[i] * st
                if want > n:
                    raise ValueError(
                        f"axis {i} has {n} points, but spatial_crop[{i}]="
                        f"{self.spatial_crop[i]} at spatial_stride[{i}]={st} "
                        f"needs {want}. Lower the crop or the stride on that "
                        f"axis (a scalar spatial_stride applies to both).")
            lo = (n - want) // 2
            out.append(slice(lo, lo + want, st))
        return tuple(out)

    @staticmethod
    def _read_scalars(path: str, keys: Sequence[str]) -> Dict[str, float]:
        import h5py
        out: Dict[str, float] = {}
        with h5py.File(path, "r") as f:
            for grp in ("scalars", "parameters"):
                if grp not in f:
                    continue
                for k in f[grp]:
                    v = np.asarray(f[grp][k])
                    if v.size == 1:
                        out[k] = float(v.ravel()[0])
            for k, v in f.attrs.items():
                if np.ndim(v) == 0 and isinstance(v, (int, float, np.number)):
                    out.setdefault(str(k), float(v))
        missing = [k for k in keys if k not in out]
        if missing:
            raise KeyError(f"{os.path.basename(path)} has no scalar(s) "
                           f"{missing}; available: {sorted(out)}")
        return out

    @staticmethod
    def _read_time(path: str) -> Optional[np.ndarray]:
        import h5py
        with h5py.File(path, "r") as f:
            if "dimensions" in f and "time" in f["dimensions"]:
                return np.asarray(f["dimensions"]["time"]).ravel().astype(float)
        return None

    def _infer_dt(self) -> float:
        t = self._time_axis
        if t is None or len(t) < 2:
            raise ValueError(
                f"{self.data_path} has no /dimensions/time, so dt cannot be "
                f"inferred. Set `data.dt` explicitly — tau is h * dt / t_scale "
                f"and a wrong dt makes every reported time meaningless.")
        d = np.diff(t)
        if float(d.std() / max(abs(d.mean()), 1e-30)) > 1e-3:
            print(f"[well] WARNING: /dimensions/time is not uniformly spaced "
                  f"(dt varies by {100 * d.std() / abs(d.mean()):.2f}%). "
                  f"Using the median. Horizons in FRAMES stay exact; horizons "
                  f"in seconds are approximate.")
        return float(np.median(d)) * self.frame_stride

    def _report(self, n_dropped: int):
        from collections import Counter
        print(f"[well] {self.data_path}")
        print(f"       {len(self.files)} file(s), {self.n_traj} trajectories"
              + (f" ({n_dropped} excluded)" if n_dropped else ""))
        print(f"       raw spatial {self._raw_spatial} -> "
              f"{self.H} x {self.W}  (crop={self.spatial_crop}, "
              f"stride={self.spatial_stride})")
        print(f"       T {self.T_raw} -> {self.T}  "
              f"(traj_cut={self.traj_cut}, frame_stride={self.frame_stride})  "
              f"dt={self.dt:g}")
        print(f"       channels {self.channel_names}")
        c = Counter(self.traj_split)
        print(f"       official split assignment: {dict(c)}")
        if "unknown" in c:
            print("       [!] files were found flat, with no train/valid/test "
                  "directory. The official split is therefore unavailable; "
                  "scripts/prepare_split.py will fall back to stratifying on "
                  "the file parameters.")

    # -- handle management (fork- and spawn-safe) --------------------------
    #
    # The cache is keyed on (pid, file index), not file index alone, and that
    # is not defensive programming -- it is required.
    #
    # `DataLoader(num_workers > 0)` on Linux FORKS.  Forking does not call
    # `__getstate__`, so a worker inherits whatever the parent had open, and an
    # inherited HDF5 handle is unusable: libhdf5 keeps process-global state
    # (open-file table, free-space manager, metadata cache) that is duplicated
    # rather than shared by fork, and two processes driving the same handle
    # produce silent wrong reads before they produce a crash.
    #
    # The parent does hold open handles at fork time in at least one real code
    # path: `scripts/evaluate_horizon.py` builds the eval loader first, then
    # fits the climatology baselines by reading through a second store IN THE
    # MAIN PROCESS, and only afterwards iterates the loader -- which is the
    # moment the workers fork.
    #
    # Keying on the pid makes an inherited handle invisible to the child, so it
    # opens its own.  `__getstate__` still drops handles for the spawn path,
    # where pickling does happen.
    def _h(self, fi: int):
        key = (os.getpid(), fi)
        h = self._handles.get(key)
        if h is None:
            import h5py
            # Handles from a previous pid can only be a fork remnant; the OS
            # closes them when this process exits, and touching them here would
            # be the bug this method exists to avoid.
            if self._handles and all(k[0] != key[0] for k in self._handles):
                self._handles = {}
            h = h5py.File(self.files[fi], "r")
            self._handles[key] = h
        return h

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def close(self):
        me = os.getpid()
        for (pid, _), h in list(self._handles.items()):
            if pid != me:
                continue
            try:
                h.close()
            except Exception:
                pass
        self._handles = {}

    # -- the contract ------------------------------------------------------
    def read(self, traj, t_idx, ch_idx=None) -> np.ndarray:
        fi, ti = self.index[int(traj)]
        f = self._h(fi)
        t_idx = np.asarray(t_idx, dtype=np.int64)
        phys = self.traj_cut + t_idx * self.frame_stride
        if phys.max() >= self.T_raw:
            raise IndexError(f"frame {int(phys.max())} past T_raw={self.T_raw}")

        # h5py fancy indexing needs strictly increasing, unique indices.  A
        # direct-time sample is [t0-K+1 .. t0] + [t0+h], which is increasing
        # already, but the semigroup and figure paths do ask for repeats.
        uniq, inv = np.unique(phys, return_inverse=True)
        want = (list(range(len(self.channel_names))) if ch_idx is None
                else [int(c) for c in ch_idx])

        cols = []
        for c in want:
            grp, ds, comp = self.spec[c]
            d = f[grp][ds]
            sel = (ti, uniq.tolist()) + self._slices
            if comp is not None:
                sel = sel + (comp,)
            cols.append(np.asarray(d[sel], dtype=np.float32))
        out = np.stack(cols, axis=-1)[inv]
        return np.ascontiguousarray(out, dtype=np.float32)

    def summary(self) -> str:
        return (f"WellStore({os.path.basename(self.data_path.rstrip('/'))}): "
                f"N={self.n_traj} T={self.T} H={self.H} W={self.W} "
                f"C={self.n_channels} dt={self.dt:g}")

    # -- metadata used by scripts/prepare_split.py -------------------------
    @property
    def official_split(self) -> List[str]:
        """'train' | 'val' | 'test' | 'unknown', one entry per trajectory."""
        return list(self.traj_split)

    def traj_params(self, traj: int) -> Dict[str, float]:
        fi, _ = self.index[int(traj)]
        return dict(self.params_by_file[fi])

    def param_label(self, traj: int, keys: Sequence[str] = ()) -> str:
        """A stable stratification label: the file's parameter tuple."""
        p = self.traj_params(traj)
        keys = list(keys) or sorted(p)
        if not keys:
            fi, _ = self.index[int(traj)]
            return os.path.basename(self.files[fi])
        return "|".join(f"{k}={p[k]:g}" for k in keys if k in p)

    def file_of(self, traj: int) -> str:
        """Split-qualified file key, e.g. "train/gray_scott_gliders.hdf5".

        Qualified rather than bare because The Well reuses one file name per
        operating point across train/valid/test; see `_resolve_exclusions`.
        """
        return self.traj_file[int(traj)]

    def basename_of(self, traj: int) -> str:
        return self.traj_file[int(traj)].split("/", 1)[-1]

    def realization_of(self, traj: int) -> int:
        return int(self.index[int(traj)][1])
