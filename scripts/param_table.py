#!/usr/bin/env python
"""Print the true parameter counts, straight from the configs.

Every parameter figure in the paper was reconstructed analytically and they are
all wrong by a factor of about two: the training logs report 8,414,786 for
AR-FNO on Gray-Scott where the paper claims 16.80M. The FiLM/time-embedding
difference (150,912 = 4 x 33,024 + 18,816) is right, so the block count and
conditioning width in the paper are right; the mode counts, and therefore every
absolute size, are not.

This builds each model from each config and reports what the code actually
constructs, so the paper can be corrected from the source of truth.

    python scripts/param_table.py --configs configs/gray_scott.yaml configs/cylinder.yaml \
        configs/rayleigh_benard.yaml configs/lifted_h2.yaml configs/base.yaml --latex
"""
from __future__ import annotations
import argparse, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--variants", nargs="+", default=["ar_fno_r", "dt_fno"])
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()

    from common import load_config, resolve_variant      # noqa
    from data import build_eval_loader                   # noqa
    from models import build_model                       # noqa

    rows = []
    for cfg_path in a.configs:
        for v in a.variants:
            try:
                cfg = load_config(cfg_path)
                cfg = resolve_variant(cfg, v) if callable(globals().get("resolve_variant")) else cfg
                cfg.setdefault("meta", {})["model_variant"] = v
                ds = cfg.get("dataset_name", "realpde_combustion")
                _, info = build_eval_loader(ds, cfg, subset="val", horizons=[1])
                info["K"] = int(cfg.get("history_len", 4))
                m = build_model(cfg["meta"]["model_variant"], cfg,
                                {"C": info["C"], "K": info["K"]})
                n = sum(p.numel() for p in m.parameters())
                rows.append((ds, v, cfg.get("width"), cfg.get("modes1"),
                             cfg.get("modes2"), cfg.get("n_layers"), n))
                print(f"  {ds:<22}{v:<12}width {cfg.get('width')}  "
                      f"modes {cfg.get('modes1')}x{cfg.get('modes2')}  "
                      f"blocks {cfg.get('n_layers')}  params {n:,}")
            except Exception as e:                        # noqa: BLE001
                print(f"  {cfg_path} / {v}: {type(e).__name__}: {e}", file=sys.stderr)

    if a.latex and rows:
        print("\n\\begin{tabular}{lccrr}\n\\toprule")
        print("dataset & modes & blocks & AR-FNO-R & DT-FNO \\\\\n\\midrule")
        by = {}
        for ds, v, w, m1, m2, nl, n in rows:
            by.setdefault((ds, m1, m2, nl), {})[v] = n
        for (ds, m1, m2, nl), d in by.items():
            ar = d.get("ar_fno_r"); dt = d.get("dt_fno")
            f = lambda x: f"${x/1e6:.2f}$\\,M" if x else "---"
            print(f"{ds} & ${m1}\\times{m2}$ & {nl} & {f(ar)} & {f(dt)} \\\\")
        print("\\bottomrule\n\\end{tabular}")

if __name__ == "__main__":
    main()
