"""Per-channel spectral error from existing horizon_metrics.json. No re-run.

Figure 7 plots the spectrum AVERAGED over channels, which is dominated by the
largest-magnitude channel. The E_spec scalar averages the per-channel RATIOS
with equal weight. The two aggregate differently, so a figure where two curves
sit on top of each other can accompany scalars that differ 10x. Read this table
before quoting either.
"""
import json, sys
for p in sys.argv[1:]:
    r = json.load(open(p))
    sp = r.get("spectral")
    if not sp:
        print(f"{p}: no spectral block (was it run with --light?)"); continue
    hs, names = sp["horizons"], r["channel_names"]
    show = [h for h in (1, 8, 32, 128) if h in hs]
    print(f"\n=== {r['meta']['exp_name']} ===")
    print(f"{'channel':<32}" + "".join(f"{'h='+str(h):>10}" for h in show))
    print("-" * (32 + 10 * len(show)))
    for c, n in enumerate(names):
        print(f"{n:<32}" + "".join(
            f"{sp['E_spec_channel'][hs.index(h)][c]:>10.3f}" for h in show))
    print(f"{'MEAN (the reported E_spec)':<32}" + "".join(
        f"{sp['E_spec'][hs.index(h)]:>10.3f}" for h in show))
