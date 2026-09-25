"""Print per-channel E(h) from an existing horizon_metrics.json. No re-run needed."""
import json, sys
paths = sys.argv[1:]
for p in paths:
    r = json.load(open(p))
    hs, names, E = r["horizons"], r["channel_names"], r["E_channel"]
    show = [h for h in (1, 8, 32, 128, 256) if h in hs]
    print(f"\n=== {r['meta']['exp_name']} ({r['meta']['model_name']}) ===")
    print(f"{'channel':<32}" + "".join(f"{'h='+str(h):>9}" for h in show))
    print("-" * (32 + 9 * len(show)))
    rows = []
    for c, n in enumerate(names):
        vals = [E[hs.index(h)][c] for h in show]
        rows.append((n, vals))
        print(f"{n:<32}" + "".join(f"{v:>9.3f}" for v in vals))
    # what the headline would be without a given channel
    for drop in ("Absolute_Pressure",):
        if drop in names:
            keep = [v for n, v in rows if n != drop]
            print(f"\n{'E_field (all channels)':<32}" +
                  "".join(f"{r['E_field'][hs.index(h)]:>9.3f}" for h in show))
            print(f"{'E_field without '+drop:<32}" +
                  "".join(f"{sum(v[i] for v in keep)/len(keep):>9.3f}"
                          for i in range(len(show))))
