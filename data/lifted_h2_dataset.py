"""
Lifted Hydrogen Jet Flame Dataset (BLASTNet, Sharma et al. 2024).

Reference:
    Web:    https://blastnet.github.io/diluted_partially_premixed_h2air_lifted_flame
    Kaggle: https://www.kaggle.com/code/sharmapushan/liftedh2-re5000-browsedata
    Paper:  Sharma et al., Combust. Flame, 2025.
            https://doi.org/10.1016/j.combustflame.2025.114190

Directory layout (one folder per jet Reynolds number):
    <root>/
        hydrogen-jet-5000/
            chem_thermo_tran/
            data/
                UX_ms-1_id0000.dat
                UX_ms-1_id0001.dat
                ...
                UY_ms-1_idXXXX.dat
                T_K_idXXXX.dat
                YH2O_idXXXX.dat
                YOH_idXXXX.dat
                ... (other variables we ignore)
            grid/
                X_m.dat
                Y_m.dat
            info.json
        hydrogen-jet-6000/
        hydrogen-jet-7000/
        hydrogen-jet-7500/   <-- TEST
        hydrogen-jet-8000/
        hydrogen-jet-9000/
        hydrogen-jet-10000/
        hydrogen-jet-11000/

Per-snapshot raw file format (BLASTNet convention):
    little-endian float32, flat array of length Nx*Ny = 1600*2000 = 3,200,000,
    reshaped to (Nx, Ny) = (1600, 2000) in row-major (C) order.

Pipeline:
    1.  Read raw .dat files and stack into (T, H_raw, W_raw, C_raw) per case.
    2.  Down-sample 1600x2000 -> 160x200 via 10x10 block-mean.
        (block-mean is conservation-preserving; better than nearest / linear zoom
         for combustion fields with sharp flame fronts).
    3.  Concatenate cases as separate trajectories: data is a LIST of
        (T_i, H, W, C) tensors — T_i is allowed to differ across cases.
        (Required because hydrogen-jet-11000 ships frames 20..200 only,
         giving T=181 vs the default T=201.  Each case is a separate DNS
         run and is not time-aligned with the others, so we keep each at
         its own length rather than cropping everyone to the shortest.)
    4.  Per-channel z-score normalisation:  x_c <- (x_c - mu_c) / sigma_c
        - statistics computed on TRAIN split only
        - test set uses the *same* mu/sigma  (no leakage)
        - this is mandatory because UX (~1e2) and YOH (~1e-3) span 5+ orders.
    5.  Optional log10-with-floor for trace species (YOH, YH2O) before z-score:
              y = log10( max(x, log_floor) )
        Mass fractions span [0, ~1e-1] with a long left tail at zero, which is
        bad for MSE-style losses — log10 turns it into a bounded, near-uniform
        scale before z-score.  Off by default for clean comparison.
    6.  Cache the post-pipeline tensor to disk (.npy) so re-runs are instant.

Channel layout (used everywhere):
    [UX, UY, T, YH2O, YOH]   (5 channels)

The 5 channels are returned in this fixed order; `data_info["channel_names"]`
exposes the names so the trainer can produce per-channel validation metrics.
"""

import os
import json
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

# (file_basename, display_name)
# The file_basename matches the prefix used by BLASTNet's info.json /
# data/<VAR>_id<XXXX>.dat naming convention.
CHANNELS: List[Tuple[str, str]] = [
    ("UX_ms-1", "UX"),
    ("UY_ms-1", "UY"),
    ("T_K",     "T"),
    ("YH2O",    "YH2O"),
    ("YOH",     "YOH"),
]
CHANNEL_FILE_PREFIXES: List[str] = [c[0] for c in CHANNELS]
CHANNEL_NAMES:         List[str] = [c[1] for c in CHANNELS]
N_CHANNELS = len(CHANNELS)

# Mass-fraction channels (good candidates for log-transform pre-processing).
MASS_FRACTION_CHANNEL_IDX = [3, 4]   # YH2O, YOH

DEFAULT_TRAIN_RES = [5000, 6000, 7000, 8000, 9000, 10000, 11000]
DEFAULT_TEST_RES  = [7500]


# ----------------------------------------------------------------------
# Low-level I/O
# ----------------------------------------------------------------------

def _read_dat(path: str, nx: int, ny: int) -> np.ndarray:
    """Read one BLASTNet .dat file -> (nx, ny) float32 array."""
    arr = np.fromfile(path, dtype="<f4")   # little-endian float32
    if arr.size != nx * ny:
        raise ValueError(
            f"Bad size for {path}: got {arr.size} values, expected {nx*ny} "
            f"({nx}x{ny}). Check info.json's Nxyz."
        )
    return arr.reshape(nx, ny)


def _block_mean_downsample(arr: np.ndarray, block: int) -> np.ndarray:
    """
    Downsample (H, W) -> (H//block, W//block) by block-mean averaging.
    Preserves spatial means; well-suited to conservation-law fields.
    """
    H, W = arr.shape
    if H % block != 0 or W % block != 0:
        raise ValueError(
            f"Cannot block-mean (H, W)=({H},{W}) with block={block}: "
            f"H and W must be divisible by block."
        )
    return arr.reshape(H // block, block, W // block, block).mean(axis=(1, 3))


# ----------------------------------------------------------------------
# Per-case loader
# ----------------------------------------------------------------------

def _load_one_case(
    case_dir: str,
    n_snapshots: Optional[int] = None,
    target_h: int = 160,
    target_w: int = 200,
) -> np.ndarray:
    """
    Load one Re case folder and return a (T, target_h, target_w, C) float32 array.
    """
    info_path = os.path.join(case_dir, "info.json")
    if not os.path.isfile(info_path):
        raise FileNotFoundError(f"info.json not found in {case_dir}")

    with open(info_path, "r") as f:
        info = json.load(f)

    nx_raw, ny_raw = info["global"]["Nxyz"]   # e.g. [1600, 2000]
    snapshots_total = info["global"]["snapshots"]
     
    if n_snapshots is None:
        n_snapshots = snapshots_total
    n_snapshots = min(n_snapshots, snapshots_total, len(info["local"]))

    if nx_raw % target_h != 0 or ny_raw % target_w != 0:
        raise ValueError(
            f"Downsample factors must be integers: "
            f"raw ({nx_raw},{ny_raw}) -> target ({target_h},{target_w})"
        )
    block_h = nx_raw // target_h
    block_w = ny_raw // target_w
    if block_h != block_w:
        # The data is uniformly spaced (15 µm) in both directions, so this is
        # almost always desired; keep it as a soft check rather than a hard error.
        print(f"  [warn] non-square block factors: ({block_h}, {block_w}) "
              f"in {os.path.basename(case_dir)}")
    if "hydrogen-jet-11000" in case_dir:
        #t_ = 20
        out = np.empty((n_snapshots - 20, target_h, target_w, N_CHANNELS), dtype=np.float32)
    else: 
        out = np.empty((n_snapshots, target_h, target_w, N_CHANNELS), dtype=np.float32)

    data_dir = os.path.join(case_dir, "data")
    first_id = None
    last_id = None
    
    t_ = 0
    if "hydrogen-jet-11000" in case_dir:
        t_ = 20

    for t in range(t_, n_snapshots):
        local = info["local"][t]
        # The actual file id can differ from the loop index when a case is
        # missing leading frames (e.g. hydrogen-jet-11000 ships frames 20..200,
        # so info["local"][0]["id"] == 20).  Always trust info.json's id field.
        file_id = local.get("id", local.get("time step", t))
        if first_id is None:
            first_id = file_id
        last_id = file_id

        for c, prefix in enumerate(CHANNEL_FILE_PREFIXES):
            key = f"{prefix} filename"
            if key in local:
                rel_path = local[key]
            else:
                # Fall back to the conventional filename, using the REAL id.
                rel_path = f"./data/{prefix}_id{file_id:04d}.dat"
            full_path = os.path.normpath(os.path.join(case_dir, rel_path))
            raw = _read_dat(full_path, nx_raw, ny_raw)
            # Block-mean downsample; works even when block_h != block_w.
            ds = raw.reshape(target_h, block_h, target_w, block_w).mean(axis=(1, 3))
            if "hydrogen-jet-11000" in case_dir:
                out[t - t_, :, :, c] = ds.astype(np.float32)
            else:
                out[t, :, :, c] = ds.astype(np.float32)

    if first_id != 0 or last_id != first_id + n_snapshots - 1:
        print(f"  [info] {os.path.basename(case_dir)}: loaded {n_snapshots} frames "
              f"(file ids {first_id}..{last_id})")

    return out


# ----------------------------------------------------------------------
# Cache key
# ----------------------------------------------------------------------

def _cache_key(re_list: List[int], target_h: int, target_w: int,
               channels: List[str], split: str) -> str:
    """Stable short key encoding the load configuration."""
    payload = json.dumps({
        "re": sorted(re_list),
        "h": target_h, "w": target_w,
        "ch": channels, "split": split,
    }, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()[:10]


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------

class LiftedH2Dataset(Dataset):
    """
    Lifted hydrogen jet flame spatiotemporal forecasting dataset.

    Each Re case becomes one trajectory of length T (default 201).
    Sliding-window samples along the time axis, just like KolmogorovFlowDataset.

    Output shapes (per __getitem__):
        data_input  : (input_len,  H, W, C=5)   float32
        data_target : (output_len * n_pushforward_stages, H, W, C=5)
        input_times : (input_len, 1)            float32 in [0, 1]
        output_times: (output_len, 1)
    """

    def __init__(
        self,
        data_root: str,
        re_list: List[int],
        split: str = "train",
        input_len: int = 10,
        output_len: int = 10,
        target_h: int = 160,
        target_w: int = 200,
        stride: int = 1,
        normalize: bool = True,
        log_mass_fractions: bool = False,
        log_floor: float = 1e-10,         # values below this are clamped before log10
        mean: Optional[np.ndarray] = None,    # (1,1,1,1,C) — train statistics
        std:  Optional[np.ndarray] = None,
        n_pushforward_stages: int = 1,
        cache_dir: Optional[str] = None,
        rebuild_cache: bool = False,
        n_snapshots: Optional[int] = None,    # truncate per-case (debug)
    ):
        super().__init__()
        self.data_root = data_root
        self.re_list = sorted(re_list)
        self.split = split
        self.input_len = input_len
        self.output_len = output_len
        self.total_len = input_len + output_len * n_pushforward_stages
        self.target_h = target_h
        self.target_w = target_w
        self.normalize = normalize
        self.log_mass_fractions = log_mass_fractions
        self.log_floor = log_floor
        self.data_dim = N_CHANNELS

        # ---- 1. raw assembly (with cache) ----
        if cache_dir is None:
            cache_dir = os.path.join(data_root, "_cache")
        os.makedirs(cache_dir, exist_ok=True)
        key = _cache_key(self.re_list, target_h, target_w, CHANNEL_NAMES, split)
        # NOTE: .npz (not .npy) — supports variable-length trajectories.
        cache_path = os.path.join(
            cache_dir,
            f"liftedH2_{split}_h{target_h}w{target_w}_{key}.npz",
        )

        if os.path.isfile(cache_path) and not rebuild_cache:
            print(f"[LiftedH2:{split}] Loading cached tensor: {cache_path}")
            loaded = np.load(cache_path)
            # Restore in re_list order:  traj_0, traj_1, ...
            traj_arrays = [loaded[f"traj_{i}"] for i in range(len(self.re_list))]
        else:
            print(f"[LiftedH2:{split}] Building from raw .dat files "
                  f"(this may take a few minutes)...")
            traj_arrays = []
            for re in self.re_list:
                case_dir = os.path.join(data_root, f"hydrogen-jet-{re}")
                if not os.path.isdir(case_dir):
                    raise FileNotFoundError(f"Case folder missing: {case_dir}")
                print(f"  Re={re}: {case_dir}")
                a = _load_one_case(case_dir, n_snapshots, target_h, target_w)
                traj_arrays.append(a)
                print(f"    -> shape {a.shape}, "
                      f"min/max per ch: "
                      + ", ".join(
                          f"{nm}=[{a[..., c].min():.3g},{a[..., c].max():.3g}]"
                          for c, nm in enumerate(CHANNEL_NAMES)
                      ))

            # Variable-length support: cases may differ in T (e.g. Re=11000
            # ships 181 frames vs 201 in the others).  Save each separately.
            np.savez(cache_path,
                     **{f"traj_{i}": a for i, a in enumerate(traj_arrays)})
            T_per = [a.shape[0] for a in traj_arrays]
            print(f"[LiftedH2:{split}] Cached -> {cache_path}, "
                  f"trajectories={len(traj_arrays)}, T_per_case={T_per}")

        # Sanity: shapes must agree on H, W, C; T is allowed to differ.
        H_ref, W_ref, C_ref = traj_arrays[0].shape[1:]
        for i, a in enumerate(traj_arrays):
            assert a.shape[1:] == (H_ref, W_ref, C_ref), (
                f"trajectory {i}: shape {a.shape} disagrees on (H,W,C) "
                f"with reference {(H_ref, W_ref, C_ref)}"
            )
        assert C_ref == N_CHANNELS, f"unexpected channel count {C_ref}"

        T_per_case = [a.shape[0] for a in traj_arrays]
        if len(set(T_per_case)) > 1:
            details = ", ".join(f"Re{re}:T={T}"
                                for re, T in zip(self.re_list, T_per_case))
            print(f"  [info] Trajectory lengths differ — {details}.  "
                  f"Each case keeps its own length (no cropping).")

        # ---- 2. (optional) log-transform mass-fraction channels ----
        if log_mass_fractions:
            for i, a in enumerate(traj_arrays):
                a = a.copy()      # was a view from np.load if cached; make writable
                for c in MASS_FRACTION_CHANNEL_IDX:
                    a[..., c] = np.log10(np.maximum(a[..., c], log_floor)).astype(np.float32)
                traj_arrays[i] = a

        # ---- 3. per-channel z-score ----
        if normalize:
            if mean is not None and std is not None:
                self.mean = np.asarray(mean, dtype=np.float32)
                self.std  = np.asarray(std,  dtype=np.float32)
            else:
                # Cell-weighted statistics across the UNION of all trajectories
                # (handles variable T correctly; degenerates to a normal mean
                # when all T are equal).
                sum_x  = np.zeros(N_CHANNELS, dtype=np.float64)
                sum_x2 = np.zeros(N_CHANNELS, dtype=np.float64)
                n_cells = 0
                for a in traj_arrays:
                    flat = a.reshape(-1, N_CHANNELS).astype(np.float64)
                    sum_x  += flat.sum(axis=0)
                    sum_x2 += (flat ** 2).sum(axis=0)
                    n_cells += flat.shape[0]
                mean_arr = (sum_x / n_cells).astype(np.float32)
                var_arr  = (sum_x2 / n_cells - mean_arr ** 2)
                std_arr  = np.sqrt(np.maximum(var_arr, 0.0)).astype(np.float32) + 1e-8
                self.mean = mean_arr.reshape(1, 1, 1, 1, N_CHANNELS)
                self.std  = std_arr .reshape(1, 1, 1, 1, N_CHANNELS)

            # Apply normalization in-place on each trajectory.
            mean_b = self.mean.reshape(1, 1, 1, N_CHANNELS)   # broadcast vs (T,H,W,C)
            std_b  = self.std .reshape(1, 1, 1, N_CHANNELS)
            for i in range(len(traj_arrays)):
                traj_arrays[i] = ((traj_arrays[i] - mean_b) / std_b).astype(np.float32)
        else:
            self.mean = np.zeros((1, 1, 1, 1, N_CHANNELS), dtype=np.float32)
            self.std  = np.ones ((1, 1, 1, 1, N_CHANNELS), dtype=np.float32)

        # ---- 4. final torch tensors & per-trajectory sliding windows ----
        # List of (T_i, H, W, C) tensors — variable T_i across i.
        self.data = [torch.from_numpy(a).float() for a in traj_arrays]
        self.T_per_case = T_per_case

        self.samples = []
        for i, a in enumerate(self.data):
            T_i = a.shape[0]
            for t_start in range(0, T_i - self.total_len + 1, stride):
                self.samples.append((i, t_start))

        # ---- diagnostics ----
        total_frames = sum(T_per_case)
        print(f"[LiftedH2:{split}] {len(self.data)} trajectories, "
              f"total frames {total_frames}, samples {len(self.samples)} "
              f"(input_len={input_len}, output_len={output_len}, "
              f"n_pushforward_stages={n_pushforward_stages}, stride={stride})")
        if normalize:
            mu = self.mean.reshape(-1)
            sd = self.std.reshape(-1)
            print("  Per-channel stats (pre-normalisation):")
            for c, nm in enumerate(CHANNEL_NAMES):
                print(f"    {nm:6s}  mean={mu[c]:+.4e}  std={sd[c]:.4e}")

    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        traj_idx, t_start = self.samples[idx]
        # self.data is a list of (T_i, H, W, C) tensors; trajectories may
        # have different T_i (e.g. Re=11000 has 181 instead of 201).
        window = self.data[traj_idx][t_start: t_start + self.total_len]

        data_input  = window[: self.input_len]
        data_target = window[self.input_len:]

        input_times  = torch.linspace(0, 1, self.input_len ).unsqueeze(-1)
        output_times = torch.linspace(0, 1, self.output_len).unsqueeze(-1)

        return {
            "data_input":   data_input,
            "data_target":  data_target,
            "input_times":  input_times,
            "output_times": output_times,
        }

    # ------------------------------------------------------------------

    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """Invert per-channel z-score (and log1p if enabled) for any tensor
        whose last dim is the channel dim, shape (..., C)."""
        mean = torch.from_numpy(self.mean).float().to(data.device)
        std  = torch.from_numpy(self.std ).float().to(data.device)
        while mean.dim() > data.dim():
            mean = mean.squeeze(0)
            std  = std.squeeze(0)
        out = data * std + mean
        if self.log_mass_fractions:
            out = out.clone()
            for c in MASS_FRACTION_CHANNEL_IDX:
                out[..., c] = torch.pow(10.0, out[..., c])   # 10^x
        return out


# ----------------------------------------------------------------------
# Per-channel evaluation helper (used by the trainer / eval scripts)
# ----------------------------------------------------------------------

@torch.no_grad()
def per_channel_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    channel_names: List[str] = CHANNEL_NAMES,
    denorm_fn=None,
) -> dict:
    """
    Compute per-channel MSE / RMSE / MAE / Rel-L2 in normalized space, and
    optionally also in physical units if `denorm_fn` is given.

    pred / target : (B, T, H, W, C)  in normalized space
    Returns flat dict with keys like:
        ch/UX/mse, ch/UX/rmse, ch/UX/mae, ch/UX/rel_l2
        ch/UX/mse_phys, ch/UX/rmse_phys, ...   (if denorm_fn given)
    """
    out = {}
    C = pred.shape[-1]
    assert len(channel_names) == C, "channel_names length must match C"

    def _stats(p, t, suffix=""):
        for c, name in enumerate(channel_names):
            pc = p[..., c]
            tc = t[..., c]
            diff = pc - tc
            mse = (diff ** 2).mean().item()
            mae = diff.abs().mean().item()
            num = torch.norm(diff.reshape(diff.shape[0], -1), dim=-1).sum().item()
            den = torch.norm(tc.reshape(tc.shape[0], -1),     dim=-1).sum().item()
            rel = num / max(den, 1e-12)
            out[f"ch/{name}/mse{suffix}"]    = mse
            out[f"ch/{name}/rmse{suffix}"]   = mse ** 0.5
            out[f"ch/{name}/mae{suffix}"]    = mae
            out[f"ch/{name}/rel_l2{suffix}"] = rel

    _stats(pred, target, suffix="")
    if denorm_fn is not None:
        _stats(denorm_fn(pred), denorm_fn(target), suffix="_phys")
    return out


# ----------------------------------------------------------------------
# Dataloader factory
# ----------------------------------------------------------------------

def create_lifted_h2_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader, dict]:
    """Create train/test dataloaders for the lifted hydrogen jet dataset."""
    from torch.utils.data.distributed import DistributedSampler

    distributed = config.get("distributed", False)

    data_root = config["data_root"]
    train_res = config.get("train_re_list", DEFAULT_TRAIN_RES)
    test_res  = config.get("test_re_list",  DEFAULT_TEST_RES)
    target_h  = config.get("x_num",   160)
    target_w  = config.get("y_num",   200)
    cache_dir = config.get("cache_dir", None)
    rebuild_cache = config.get("rebuild_cache", False)
    log_mass_fractions = config.get("log_mass_fractions", False)
    log_floor = config.get("log_floor", 1e-10)
    n_snapshots = config.get("n_snapshots", None)

    train_dataset = LiftedH2Dataset(
        data_root=data_root,
        re_list=train_res,
        split="train",
        input_len=config.get("input_len", 10),
        output_len=config.get("output_len", 10),
        target_h=target_h, target_w=target_w,
        stride=config.get("stride_train", 1),
        normalize=config.get("normalize", True),
        log_mass_fractions=log_mass_fractions,
        log_floor=log_floor,
        n_pushforward_stages=config.get("n_pushforward_stages", 1),
        cache_dir=cache_dir, rebuild_cache=rebuild_cache,
        n_snapshots=n_snapshots,
    )
    test_dataset = LiftedH2Dataset(
        data_root=data_root,
        re_list=test_res,
        split="test",
        input_len=config.get("input_len", 10),
        output_len=config.get("output_len", 10),
        target_h=target_h, target_w=target_w,
        stride=config.get("stride_test", 10),
        normalize=config.get("normalize", True),
        log_mass_fractions=log_mass_fractions,
        log_floor=log_floor,
        mean=train_dataset.mean,    # CRUCIAL: reuse train statistics
        std=train_dataset.std,
        n_pushforward_stages=1,
        cache_dir=cache_dir, rebuild_cache=rebuild_cache,
        n_snapshots=n_snapshots,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True ) if distributed else None
    test_sampler  = DistributedSampler(test_dataset,  shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_dataset, batch_size=config.get("batch_size", 8),
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=config.get("num_workers", 4),
        pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.get("batch_size_eval", 16),
        shuffle=False, sampler=test_sampler,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )

    data_info = {
        "x_num":      target_h,
        "y_num":      target_w,
        "data_dim":   train_dataset.data_dim,
        "input_len":  config.get("input_len",  10),
        "output_len": config.get("output_len", 10),
        "train_size": len(train_dataset),
        "test_size":  len(test_dataset),
        "mean":       train_dataset.mean,
        "std":        train_dataset.std,
        "channel_names":           CHANNEL_NAMES,
        "channel_file_prefixes":   CHANNEL_FILE_PREFIXES,
        "log_mass_fractions":      log_mass_fractions,
        "denormalize":             train_dataset.denormalize,
    }
    return train_loader, test_loader, data_info


# ----------------------------------------------------------------------
# Synthetic data (smoke-test the pipeline without the real ~25 GB dataset)
# ----------------------------------------------------------------------

def generate_synthetic_lifted_h2(
    save_root: str = "./data/lifted_h2_synth/",
    re_list: List[int] = (5000, 7500, 10000),
    n_snapshots: int = 40,
    nx: int = 160, ny: int = 200,    # already-downsampled size
):
    """Write fake .dat files that match the BLASTNet folder layout, for testing."""
    os.makedirs(save_root, exist_ok=True)
    rng = np.random.default_rng(0)

    for re in re_list:
        case_dir = os.path.join(save_root, f"hydrogen-jet-{re}")
        data_dir = os.path.join(case_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        info = {
            "global": {
                "dataset_id": f"sharmapushan/hydrogen-jet-{re}",
                "Nxyz": [nx, ny],
                "snapshots": n_snapshots,
                "variables": ["RHO_kgm-3", "UX_ms-1", "UY_ms-1", "P_Pa", "T_K",
                              "YH", "YH2", "YO", "YO2", "YOH", "YH2O", "YHO2", "YH2O2"],
                "Re_jet": re,
            },
            "local": [],
        }

        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y, indexing="ij")

        for t in range(n_snapshots):
            tau = 0.05 * t
            U_jet = 154.0 * (re / 5000.0)
            UX = (U_jet * np.exp(-((Y - 0.5) / 0.1) ** 2)
                  + 5 * np.sin(4 * X + tau))
            UY = 10.0 * np.sin(6 * Y + tau) * np.cos(3 * X)
            T  = 400 + 1700 * np.exp(-((Y - 0.5) / 0.15) ** 2) * (X + 0.1) / 1.1
            YH2O = 0.15 * np.exp(-((Y - 0.5) / 0.12) ** 2) * (X + 0.05)
            YOH  = 5e-3 * np.exp(-((Y - 0.5) / 0.10) ** 2) * X * (1 - X)

            arrays = {
                "UX_ms-1": UX, "UY_ms-1": UY, "T_K": T,
                "YH2O": YH2O, "YOH": YOH,
                # also write the unused channels so info.json is honest
                "RHO_kgm-3": np.ones_like(UX) + 0.1 * rng.standard_normal(UX.shape),
                "P_Pa": 101325 + 100 * rng.standard_normal(UX.shape),
                "YH":   1e-6 * np.abs(rng.standard_normal(UX.shape)),
                "YH2":  0.65 * np.exp(-((Y - 0.5) / 0.05) ** 2) * np.exp(-X),
                "YO":   1e-6 * np.abs(rng.standard_normal(UX.shape)),
                "YO2":  0.21 * np.ones_like(UX),
                "YHO2": 1e-7 * np.abs(rng.standard_normal(UX.shape)),
                "YH2O2": 1e-8 * np.abs(rng.standard_normal(UX.shape)),
            }
            local_entry = {"id": t, "time step": t}
            for varname, arr in arrays.items():
                fname = f"{varname}_id{t:04d}.dat"
                arr.astype("<f4").tofile(os.path.join(data_dir, fname))
                local_entry[f"{varname} filename"] = f"./data/{fname}"
            info["local"].append(local_entry)

        with open(os.path.join(case_dir, "info.json"), "w") as f:
            json.dump(info, f, indent=2)
        print(f"  Wrote synthetic case: {case_dir}  ({n_snapshots} snapshots, {nx}x{ny})")

    print(f"Synthetic lifted-H2 root: {save_root}")
    return save_root


if __name__ == "__main__":
    # quick smoke test
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="./data/lifted_h2_synth")
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()

    if args.synthetic:
        generate_synthetic_lifted_h2(save_root=args.root)

    cfg = {
        "data_root": args.root,
        "train_re_list": [5000, 10000],
        "test_re_list":  [7500],
        "x_num": 160, "y_num": 200,
        "input_len": 10, "output_len": 10,
        "batch_size": 2, "num_workers": 0,
    }
    if args.synthetic:
        # synthetic generator above writes already at 160x200, so no downsample
        # NB: real dataset is 1600x2000 raw -> 160x200 here.
        pass

    train_loader, test_loader, info = create_lifted_h2_dataloaders(cfg)
    print("data_info:", {k: v for k, v in info.items() if k not in ("denormalize",)})
    batch = next(iter(train_loader))
    for k, v in batch.items():
        print(f"  {k}: {tuple(v.shape)}  dtype={v.dtype}")
