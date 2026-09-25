"""Model registry.

Three trained models (§3), all built from the same backbone so a measured gap
is attributable to the prediction scheme rather than the architecture:

    ar_fno      Model A   autoregressive; config `ar_rollout` picks -1 vs -R
    dt_fno      Model B   direct-time, time-conditioned
    sg_dt_fno   Model C   Model B; the semigroup losses live in the task

Plus two untrained sanity baselines from `models/baselines.py`.
"""

from __future__ import annotations

from typing import Dict

import torch.nn as nn

from .ar_fno import ARFNO
from .direct_fno import DirectTimeFNO, SG_DT_FNO
from .baselines import Persistence, PODDMD
from .fno import FNOBackbone, FNOBlock
from .spectral import SpectralConv2d
from .time_embedding import FourierTimeEmbedding, FiLM
from .history_encoder import HistoryEncoder, Decoder

MODEL_REGISTRY = {
    "ar_fno": "ar",
    "dt_fno": "direct",
    "sg_dt_fno": "direct",
    # Second backbone, for the architecture-robustness check. Same two classes
    # as above -- ARFNO and DirectTimeFNO -- with `backbone="unet"` swapping
    # FNOBackbone for UNetBackbone and nothing else. A crossover is a property
    # of two curves, so testing whether it is an FNO artefact needs a MATCHED
    # PAIR in the second architecture, not a lone direct model.
    "ar_unet": "ar",
    "dt_unet": "direct",
}

# ---------------------------------------------------------------------------
# Training variants
# ---------------------------------------------------------------------------
# AR-FNO-1 and AR-FNO-R are the SAME model (§23): both are `ar_fno`, and the
# difference is the training distribution, not the network. So "ar_fno_r" is a
# variant name, not a model name, and looking it up in MODEL_REGISTRY is a
# KeyError.
#
# These settings used to live only inside configs/ar_fno_r.yaml et al, which
# worked as long as every run picked its config by model. The moment one config
# is pinned for a whole dataset (CONFIG=configs/lifted_h2.yaml in
# run/07_seeds.sh) that knowledge is orphaned. This table is the single source
# of truth; the per-model configs declare `model_variant` and inherit from it
# rather than restating the values.
#
# Precedence, applied in scripts/common.resolve():
#     config file  <  variant table  <  explicit --set on the command line
MODEL_VARIANTS = {
    # --- second backbone -------------------------------------------------
    # ar_unet_r mirrors ar_fno_r EXACTLY (rollout 4, random). An AR baseline
    # trained on one-step supervision would diverge earlier for a reason that
    # has nothing to do with the architecture, and its crossover would not be
    # comparable with the one in the paper.
    "ar_unet_r": {
        "model_name": "ar_unet",
        "ar_rollout": 4,
        "ar_random_rollout": True,
        "horizon_sampling": "fixed",
        "fixed_horizon": 1,
    },
    "ar_unet": {
        "model_name": "ar_unet",
        "ar_rollout": 1,
        "ar_random_rollout": False,
        "horizon_sampling": "fixed",
        "fixed_horizon": 1,
    },
    "dt_unet": {"model_name": "dt_unet"},
    "sg_dt_unet": {"model_name": "dt_unet",
                   "use_semigroup": True, "use_identity": True},
    "ar_fno": {                       # AR-FNO-1, one-step supervision
        "model_name": "ar_fno",
        "ar_rollout": 1,
        "ar_random_rollout": False,
        "horizon_sampling": "fixed",
        "fixed_horizon": 1,
    },
    "ar_fno_r": {                     # AR-FNO-R, short-rollout supervision
        "model_name": "ar_fno",
        "ar_rollout": 4,
        "ar_random_rollout": True,
        "horizon_sampling": "fixed",
        "fixed_horizon": 1,
    },
    "dt_fno": {
        "model_name": "dt_fno",
        "use_semigroup": False,
        "use_identity": False,
    },
    "sg_dt_fno": {
        "model_name": "sg_dt_fno",
        "use_semigroup": True,
        "use_identity": True,
    },
}
MODEL_VARIANTS["ar_fno_1"] = dict(MODEL_VARIANTS["ar_fno"])


def apply_variant(flat: dict, variant: str) -> dict:
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"unknown model_variant '{variant}'. "
                         f"Available: {sorted(MODEL_VARIANTS)}")
    out = dict(flat)
    out.update(MODEL_VARIANTS[variant])
    return out


def list_variants():
    return sorted(MODEL_VARIANTS)


def _shared_kwargs(cfg: dict, data_info: dict) -> dict:
    return dict(
        n_channels=int(data_info["C"]),
        history_len=int(data_info["K"]),
        width=int(cfg.get("width", 64)),
        modes1=int(cfg.get("modes1", 16)),
        modes2=int(cfg.get("modes2", 16)),
        n_layers=int(cfg.get("n_layers", 4)),
        decoder_hidden=int(cfg.get("decoder_hidden", 128)),
        encoder_kernel=int(cfg.get("encoder_kernel", 1)),
        padding=int(cfg.get("padding", 8)),
        padding_mode=str(cfg.get("padding_mode", "replicate")),
        residual=bool(cfg.get("residual", True)),
        layer_scale=float(cfg.get("layer_scale", 0.1)),
        predict_delta=bool(cfg.get("predict_delta", True)),
        norm=bool(cfg.get("block_norm", False)),
        act=str(cfg.get("act", "gelu")),
    )


def build_model(name: str, cfg: dict, data_info: dict) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    kw = _shared_kwargs(cfg, data_info)
    kw["backbone"] = "unet" if name.endswith("_unet") else "fno"
    kw["unet_base_width"] = int(cfg.get("unet_base_width", 80))
    kw["unet_depth"] = int(cfg.get("unet_depth", 3))
    if MODEL_REGISTRY[name] == "ar":
        return ARFNO(**kw)
    return DirectTimeFNO(
        time_embed_dim=int(cfg.get("time_embed_dim", 128)),
        time_bands=int(cfg.get("time_bands", 8)),
        time_embed_mode=str(cfg.get("time_embed_mode", "fourier")),
        time_cond=str(cfg.get("time_cond", "film")),
        **kw)


def model_kind(name: str) -> str:
    """'direct' or 'ar' — tells the data builder which dataset to construct."""
    return MODEL_REGISTRY[name]


def count_params(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    by_part = {}
    for part in ("encoder", "backbone", "decoder", "time_embed", "injector"):
        m = getattr(model, part, None)
        if isinstance(m, nn.Module):
            by_part[part] = sum(p.numel() for p in m.parameters())
    return {"total": total, **by_part}


def list_models():
    return list(MODEL_REGISTRY)
