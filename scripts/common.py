"""
Shared plumbing for every script: sys.path, config loading, seeding, DDP setup.

Config files are nested for readability (meta / data / model / training /
experiment) and flattened into one dict before use, matching the convention in
the user's existing `train.py`.  Flattening means a key set in `data:` is
visible to the model builder and vice versa, which is deliberate: `history_len`
is a property of the problem, not of one subsystem, and having it appear twice
is how the two silently diverge.

`--set a.b=c` applies a dotted override to the nested config before flattening,
so an ablation sweep needs no config file per arm:

    python scripts/train.py --config configs/sg_dt_fno.yaml \
        --set training.lambda_sg=0.2 --set experiment.exp_name=sg_lam0.2
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Any, Dict, List, Optional

# repo root on the path so `data`, `models`, ... import as top-level packages
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import torch        # noqa: E402
import yaml         # noqa: E402

REPO_ROOT = _ROOT


def load_config(path: str) -> dict:
    with open(path) as fp:
        cfg = yaml.safe_load(fp) or {}
    base = cfg.pop("_base_", None)
    if base:                                  # one level of config inheritance
        base_path = base if os.path.isabs(base) else os.path.join(
            os.path.dirname(os.path.abspath(path)), base)
        merged = load_config(base_path)
        cfg = deep_update(merged, cfg)
    return cfg


def deep_update(base: dict, new: dict) -> dict:
    out = dict(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def apply_overrides(cfg: dict, overrides: Optional[List[str]]) -> dict:
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got '{item}'")
        key, raw = item.split("=", 1)
        val = yaml.safe_load(raw)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return cfg


def flatten(cfg: dict) -> dict:
    flat: Dict[str, Any] = {}
    for section, values in cfg.items():
        if isinstance(values, dict):
            flat.update(values)
        else:
            flat[section] = values
    return flat


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed(flat: dict) -> bool:
    """torchrun sets LOCAL_RANK; without it we run single-process."""
    if "LOCAL_RANK" not in os.environ:
        flat["distributed"] = False
        flat.setdefault("device",
                        "cuda" if torch.cuda.is_available() else "cpu")
        return False
    import torch.distributed as dist
    lr = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(lr)
        dist.init_process_group(backend="nccl")
        flat["device"] = f"cuda:{lr}"
    else:
        # CPU gloo path: not for training, but it lets the DDP code path be
        # exercised without a GPU, which is where multi-rank bugs hide.
        dist.init_process_group(backend="gloo")
        flat["device"] = "cpu"
    flat["distributed"] = True
    return True


def is_master() -> bool:
    import torch.distributed as dist
    return (not (dist.is_available() and dist.is_initialized())
            or dist.get_rank() == 0)


def cleanup_distributed():
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True)
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE",
                   help="dotted override, e.g. training.lr=3e-4")
    p.add_argument("--seed", type=int, default=None)
    return p


def resolve(args) -> dict:
    overrides = getattr(args, "overrides", None) or []
    cfg = apply_overrides(load_config(args.config), overrides)
    flat = flatten(cfg)

    # `model_variant` expands the AR-FNO-1 / AR-FNO-R style training variants
    # (see models.MODEL_VARIANTS). Command-line --set is re-applied afterwards so
    # an explicit override always wins over the variant table, which in turn
    # wins over the config file.
    variant = flat.get("model_variant")
    if variant:
        from models import apply_variant
        flat = apply_variant(flat, str(variant))
        for item in overrides:
            key, raw = item.split("=", 1)
            leaf = key.split(".")[-1]
            if leaf != "model_variant":
                flat[leaf] = yaml.safe_load(raw)

    if getattr(args, "seed", None) is not None:
        flat["seed"] = args.seed
    flat.setdefault("seed", 42)
    return flat


def json_default(o):
    """numpy/torch scalars are not JSON-serialisable; make them so."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if torch.is_tensor(o):
        return o.detach().cpu().tolist()
    if hasattr(o, "__dict__"):
        return {k: v for k, v in vars(o).items() if not k.startswith("_")}
    return str(o)
