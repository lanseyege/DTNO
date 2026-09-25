"""
RealPDEBench Hugging Face (Arrow) backend — zero-copy, frame-random.

RealPDEBench v2 ships each scenario as a `datasets.save_to_disk` directory:

    <root>/cylinder/hf_dataset/numerical/
        data-00000-of-00092.arrow  ...  one shard, one complete trajectory
        dataset_info.json  state.json
    <root>/cylinder/hf_dataset/{train,val,test}_index_numerical.json

and each Arrow *row* holds a complete trajectory with the fields stored as raw
little-endian float32 blobs:

    sim_id  (str)              e.g. "10031.h5" — the Reynolds number
    u, v, p (bytes)            (shape_t, shape_h, shape_w) float32
    vo      (bytes)            vorticity, a function of u and v
    x, y    (bytes)            (shape_h, shape_w) coordinate grids
    t       (bytes)            (shape_t,) timestamps
    shape_t, shape_h, shape_w  (int)


WHY NOT THE OFFICIAL LOADER
---------------------------
`fluid_hf_dataset.CylinderHFDataset.__getitem__` does

    row = self.trajectories[traj_idx]
    u_full = np.frombuffer(row["u"], ...).reshape(3990, H, W)
    u = u_full[time_id : time_id + horizon]

which materialises **the whole 3990-frame trajectory** (130 MB per field,
390 MB for u/v/p) to hand back a window.  That is fine for their protocol, where
a sample is one contiguous window; it is fatal for ours, where a direct-time
sample is K history frames plus a single frame up to 512 steps later and the
entire affordability argument of §8 rests on that gather costing K+1 frames
regardless of h.

This store instead memory-maps the Arrow IPC files with pyarrow and takes a
numpy *view* onto the blob.  Frames are then read through the page cache: a
(K+1)-frame gather touches ~5 x H x W x 4 bytes of pages, not 390 MB, and no
copy of the dataset is written to disk.  `scripts/convert_cylinder.py` remains
available if you would rather trade ~36 GB of disk for slightly lower per-read
overhead; both routes produce identical numbers.

THREE OFFICIAL BEHAVIOURS THIS STORE DELIBERATELY DOES NOT REPRODUCE
--------------------------------------------------------------------
1. `mask_prob` — the official loader replaces the pressure channel with zeros
   with probability 0.5.  That is their sim-to-real modality-masking protocol.
   Applying it here would inject a random channel dropout into the training
   distribution of every model and into the normalisation statistics, and
   would show up as unexplained variance in E(h).
2. `sub_s_numerical = 2` — the official loader trains at 32 x 64.  We keep the
   native 64 x 128 so that resolution is not a second difference between
   Cylinder and RealPDEBench combustion.  Our numbers are therefore NOT
   comparable to the RealPDEBench leaderboard, and the paper must say so.
3. `vo` (vorticity) — a function of u and v, exactly the case §26 and the
   audit's redundancy check exist to catch (cf. `Velocity_Magnitude` in
   configs/base.yaml).  Excluded by default; add it to `arrow_fields` if you
   want it, and read step [2b] of the audit before you do.
"""

from __future__ import annotations

import glob
import json
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .store import TrajectoryStore

DEFAULT_FIELDS = ("u", "v", "p")


class ArrowTrajectoryStore(TrajectoryStore):
    """One Arrow row = one trajectory; frames are read as memory-mapped views."""

    def __init__(self,
                 data_path: str,
                 fields: Sequence[str] = DEFAULT_FIELDS,
                 dt: Optional[float] = None,
                 channel_names: Optional[Sequence[str]] = None,
                 sub_s: int = 1,
                 t_max: Optional[int] = None,
                 sim_id_regex: str = r"^(\d+)",
                 verbose: bool = True):
        self.data_path = data_path
        self.fields = list(fields)
        self.sub_s = max(1, int(sub_s))
        self.sim_id_regex = sim_id_regex
        self._tables: Dict[str, object] = {}
        self._vindex: Dict[tuple, list] = {}
        self._warned_slow = False

        self.files = sorted(glob.glob(os.path.join(data_path, "*.arrow")))
        if not self.files:
            raise FileNotFoundError(
                f"no *.arrow shards under {data_path}. Expected a "
                f"`datasets.save_to_disk` directory, e.g. "
                f"<root>/cylinder/hf_dataset/numerical/")

        # ---- index: (file, row) per trajectory ---------------------------
        self.rows: List[Tuple[str, int]] = []
        self.sim_ids: List[str] = []
        shapes: List[Tuple[int, int, int]] = []
        t_axis = None
        for path in self.files:
            tb = self._table(path)
            missing = [f for f in self.fields if f not in tb.column_names]
            if missing:
                raise KeyError(f"{os.path.basename(path)} has no column(s) "
                               f"{missing}; available: {tb.column_names}")
            n_rows = tb.num_rows
            for r in range(n_rows):
                self.rows.append((path, r))
                self.sim_ids.append(str(tb.column("sim_id")[r].as_py()))
                shapes.append((int(tb.column("shape_t")[r].as_py()),
                               int(tb.column("shape_h")[r].as_py()),
                               int(tb.column("shape_w")[r].as_py())))
            if t_axis is None and "t" in tb.column_names:
                t_axis = np.frombuffer(
                    memoryview(tb.column("t")[0].as_buffer()),
                    dtype=np.float32).astype(float)
            self._release(path)

        self.n_traj = len(self.rows)
        hs = {s[1] for s in shapes}
        ws = {s[2] for s in shapes}
        if len(hs) > 1 or len(ws) > 1:
            raise ValueError(f"trajectories disagree on spatial shape: "
                             f"H in {sorted(hs)}, W in {sorted(ws)}")
        self._raw_h, self._raw_w = shapes[0][1], shapes[0][2]
        self.native_T = [s[0] for s in shapes]

        # A ragged time axis would let a sampler draw a frame past the end of a
        # short trajectory. Truncate to the common length and say how much was
        # dropped, rather than carrying a per-trajectory T that nothing else in
        # the pipeline understands.
        T_common = min(self.native_T)
        self.T = int(min(T_common, t_max) if t_max else T_common)
        self.H = int(np.ceil(self._raw_h / self.sub_s))
        self.W = int(np.ceil(self._raw_w / self.sub_s))

        self.channel_names = (list(channel_names) if channel_names
                              else list(self.fields))
        if len(self.channel_names) != len(self.fields):
            raise ValueError(f"{len(self.channel_names)} channel names for "
                             f"{len(self.fields)} fields")

        self.dt = float(dt) if dt else self._infer_dt(t_axis)

        if verbose:
            lost = sum(n - self.T for n in self.native_T)
            print(f"[arrow] {data_path}")
            print(f"        {len(self.files)} shard(s), {self.n_traj} "
                  f"trajectories, T {min(self.native_T)}..{max(self.native_T)} "
                  f"-> {self.T}" + (f" ({lost} frames dropped to square the "
                                    f"time axis)" if lost else ""))
            print(f"        grid {self._raw_h} x {self._raw_w}"
                  + (f" -> {self.H} x {self.W} (sub_s={self.sub_s})"
                     if self.sub_s > 1 else ""))
            print(f"        channels {self.channel_names}   dt={self.dt:g}s")

    # ---------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: dict) -> "ArrowTrajectoryStore":
        return cls(
            data_path=cfg["data_path"],
            fields=cfg.get("arrow_fields", DEFAULT_FIELDS),
            dt=cfg.get("dt"),
            channel_names=cfg.get("all_channel_names"),
            sub_s=int(cfg.get("spatial_sub", 1)),
            t_max=cfg.get("t_max"),
            sim_id_regex=cfg.get("sim_id_regex", r"^(\d+)"),
            verbose=bool(cfg.get("store_verbose", True)),
        )

    # ---- pyarrow plumbing ------------------------------------------------
    #
    # Unlike `data/well.py` this cache is NOT keyed on pid, and the asymmetry
    # is deliberate.  An inherited HDF5 handle is unusable because libhdf5
    # carries process-global state that fork duplicates.  An inherited Arrow
    # memory map is a read-only mmap plus a file descriptor: fork shares it
    # correctly, every worker reads through the same page cache, and re-mapping
    # per worker would only cost address space and page faults.  Verified by
    # forking four children off a parent that had already read: identical
    # bytes, one shared mapping.
    def _table(self, path: str):
        tb = self._tables.get(path)
        if tb is None:
            import pyarrow as pa
            src = pa.memory_map(path, "rb")
            try:
                reader = pa.ipc.open_stream(src)
            except pa.ArrowInvalid:
                src.seek(0)
                reader = pa.ipc.open_file(src)
            tb = reader.read_all()
            self._tables[path] = tb
        return tb

    def _release(self, path: str):
        # Dropping the table keeps __init__ from pinning 92 mapped tables while
        # it only needs the small metadata columns. Reads re-map on demand.
        # The value index holds buffers into that mapping, so it goes too.
        self._tables.pop(path, None)
        for k in [k for k in self._vindex if k[0] == path]:
            self._vindex.pop(k, None)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_tables"] = {}
        state["_vindex"] = {}
        return state

    # ---- zero-copy value access -----------------------------------------
    #
    # The obvious way to reach one value of a binary column is
    #
    #     table.column(field)[row].as_buffer()
    #
    # and it is a trap.  Building the Scalar COPIES the whole value, so a call
    # that should hand back a pointer instead memcpys the entire field: 116 ms
    # for a 157 MB column here, and ~650 ms for the 523 MB column of a real
    # 3990-frame Cylinder trajectory.  Three channels per sample made
    # `store.read` cost ~1.3 s to deliver 2 MB -- 1.5 MB/s out of RAM, with the
    # disk idle and the whole working set in page cache.  The DataLoader scaled
    # perfectly linearly with workers, which hid it: it looked like an I/O-bound
    # dataset when it was one memcpy in the wrong place.
    #
    # The layout underneath is simple.  A binary array is three buffers --
    # validity, offsets, data -- and value `i` is `data[offsets[i]:offsets[i+1]]`.
    # `Buffer.slice` on the data buffer is a genuine zero-copy view, so the
    # whole lookup becomes two integer reads and a pointer, measured at
    # 0.001 ms and byte-identical to `as_buffer()`.
    def _value_index(self, path: str, field: str):
        """-> [(first_row, n_rows, data_buffer, offsets, chunk_offset)], cached."""
        key = (path, field)
        idx = self._vindex.get(key)
        if idx is not None:
            return idx
        import pyarrow as pa
        col = self._table(path).column(field)
        if col.null_count:
            raise ValueError(f"{os.path.basename(path)} column '{field}' has "
                             f"nulls; this store expects one blob per row")
        out, start = [], 0
        for chunk in col.chunks:
            bufs = chunk.buffers()
            if len(bufs) < 3 or bufs[1] is None or bufs[2] is None:
                raise TypeError(
                    f"column '{field}' is {chunk.type}, not a binary column "
                    f"with (validity, offsets, data) buffers")
            wide = pa.types.is_large_binary(chunk.type)
            offsets = np.frombuffer(memoryview(bufs[1]),
                                    dtype=np.int64 if wide else np.int32)
            out.append((start, len(chunk), bufs[2], offsets, chunk.offset))
            start += len(chunk)
        self._vindex[key] = out
        return out

    def _value_bytes(self, path: str, row: int, field: str) -> np.ndarray:
        """float32 view of one value. Falls back to a copy if the layout is odd."""
        try:
            for first, n_rows, data, offsets, base in self._value_index(
                    path, field):
                if first <= row < first + n_rows:
                    j = base + (row - first)
                    lo, hi = int(offsets[j]), int(offsets[j + 1])
                    return np.frombuffer(memoryview(data.slice(lo, hi - lo)),
                                         dtype=np.float32)
            raise IndexError(f"row {row} not found in {os.path.basename(path)}")
        except (TypeError, ValueError, AttributeError) as e:
            if not self._warned_slow:
                self._warned_slow = True
                print(f"[arrow] WARNING: cannot take a zero-copy view of "
                      f"column '{field}' ({e}). Falling back to "
                      f"Scalar.as_buffer(), which COPIES the whole column on "
                      f"every read -- expect ~1 s per sample. "
                      f"scripts/convert_cylinder.py avoids it entirely.")
            buf = self._table(path).column(field)[row].as_buffer()
            return np.frombuffer(memoryview(buf), dtype=np.float32)

    def _blob(self, path: str, row: int, field: str, shape) -> np.ndarray:
        arr = self._value_bytes(path, row, field)
        need = int(np.prod(shape))
        if arr.size != need:
            raise ValueError(
                f"{os.path.basename(path)} row {row} field '{field}': "
                f"{arr.size} float32 values for declared shape {tuple(shape)} "
                f"({need}). The blob is not raw little-endian float32, so this "
                f"store cannot view it; use scripts/convert_cylinder.py.")
        return arr.reshape(shape)

    def _infer_dt(self, t_axis) -> float:
        if t_axis is None or len(t_axis) < 2:
            raise ValueError(
                f"{self.data_path} exposes no 't' column, so dt cannot be "
                f"inferred. Set `data.dt` explicitly (RealPDEBench Cylinder is "
                f"2.5e-3 s, i.e. the 400 Hz PIV sampling rate).")
        return float(np.median(np.diff(t_axis)))

    # ---- the contract ----------------------------------------------------
    def read(self, traj, t_idx, ch_idx=None) -> np.ndarray:
        path, row = self.rows[int(traj)]
        t_idx = np.asarray(t_idx, dtype=np.int64)
        if t_idx.max() >= self.T:
            raise IndexError(f"frame {int(t_idx.max())} past T={self.T}")
        shape = (self.native_T[int(traj)], self._raw_h, self._raw_w)
        want = (range(len(self.fields)) if ch_idx is None
                else [int(c) for c in ch_idx])
        s = self.sub_s
        cols = []
        for c in want:
            view = self._blob(path, row, self.fields[c], shape)
            block = view[t_idx]                       # copies only these frames
            if s > 1:
                block = block[:, ::s, ::s]
            cols.append(np.asarray(block, dtype=np.float32))
        return np.ascontiguousarray(np.stack(cols, axis=-1), dtype=np.float32)

    def summary(self) -> str:
        return (f"ArrowTrajectoryStore({os.path.basename(self.data_path.rstrip('/'))}): "
                f"N={self.n_traj} T={self.T} H={self.H} W={self.W} "
                f"C={self.n_channels} dt={self.dt:g}s")

    # ---- metadata used by scripts/prepare_split.py -----------------------
    def sim_id(self, traj: int) -> str:
        return self.sim_ids[int(traj)]

    def sim_param(self, traj: int) -> Optional[float]:
        """The number in the sim_id. On Cylinder this is the Reynolds number."""
        m = re.search(self.sim_id_regex, self.sim_ids[int(traj)])
        return float(m.group(1)) if m else None

    def official_index(self, index_dir: str, dataset_type: str = "numerical"
                       ) -> Dict[str, List[dict]]:
        """Load RealPDEBench's own {split}_index_{type}.json files.

        Read the docstring of scripts/prepare_split.py before using these: the
        official split shares trajectories across train / val / test and is a
        WINDOW split, which is precisely the leak §9 exists to prevent.
        """
        out: Dict[str, List[dict]] = {}
        for split in ("train", "val", "test"):
            p = os.path.join(index_dir, f"{split}_index_{dataset_type}.json")
            if os.path.exists(p):
                with open(p) as fp:
                    out[split] = json.load(fp)
        return out
