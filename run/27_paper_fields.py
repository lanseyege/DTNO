#!/usr/bin/env bash
# Regenerate the three qualitative-field figures (scripts/plot_fields.py) with
# titles off, a dataset tag inside the first panel, and the paper's model names
# as column headers. Installs them into the LaTeX project and prints the
# caption material each figure now needs.
#
#   bash run/27_paper_fields.sh /path/to/DTNO_ICLR2026/figures
#
# CHECKPOINT TAGS ARE THE THING TO GET RIGHT
# ------------------------------------------
# Run tags are `<dataset-prefix>_<model>_s<seed>`, and the prefix is EMPTY for
# the original RealPDEBench runs and set for everything since. So
# `checkpoints/ar_fno_r_s0` is the COMBUSTION run no matter which --config is
# passed beside it, and pointing the Lifted H2 config at it loads 13-channel,
# 16x16-mode weights into a 5-channel, 16x20 model.
#
# That particular mistake crashes, which makes it the harmless version. Three
# of the five datasets share 16x16 modes, so a wrong tag that happens to agree
# on the channel count loads SILENTLY under `strict=False` and yields a figure
# captioned with the wrong trajectory. scripts/plot_fields.py now compares the
# checkpoint's recorded `config` and `data_info` against the config in use and
# refuses before building the model; this script sets the prefixes so it should
# never have to. Override if your tags differ:
#
#   LH2_PREFIX=lh2_ COMB_PREFIX= CYL_PREFIX=cyl_ SEED=s0 bash run/27_...
#
# The three figures have DIFFERENT rows, and that is correct rather than a
# configuration to unify --- each dataset has its own state:
#
#   B1 Lifted H2      Temperature [K] · OH (reaction-zone mask) · UX
#   B2 RealPDEBench   Temperature [K] · OH · Heat release · Velocity[i]
#   A2 Cylinder       u · v · pressure
#
# The row selection is automatic: plot_fields.py looks for combustion channels
# and, finding none, falls back to the dataset's own channels and omits the
# reaction-zone contour. Check the console line it prints
# ("reaction-zone mask channel: ...") before using any output --- a regex that
# silently resolves to the wrong channel once drew the OH field under a "Heat
# release" heading, and nothing in the image gives that away.
#
# Because the rows differ, they cannot identify the figure the way a shared
# axis would, which is why each gets a --label.
set -euo pipefail

PAPER_FIGS=${1:-}
OUT=${OUT:-./results/figures_paper/fields}
H=${H:-128}
CK=${CK:-./checkpoints}
SEED=${SEED:-s0}

# `${VAR-default}` rather than `${VAR:-default}`: COMB_PREFIX is legitimately
# EMPTY, and `:-` would silently replace an intentional empty string.
LH2_PREFIX=${LH2_PREFIX-lh2_}
COMB_PREFIX=${COMB_PREFIX-}
CYL_PREFIX=${CYL_PREFIX-cyl_}

available() {
  echo "    available checkpoint dirs:"
  ls -1 "$CK" 2>/dev/null | sed 's/^/      /' | head -40
}

run_fields() {   # run_fields <config> <prefix> <label> <tag> <anchor>
  local cfg="$1" pre="$2" label="$3" tag="$4" anchor="$5"
  local a="$CK/${pre}ar_fno_r_${SEED}/best_model.pth"
  local d="$CK/${pre}dt_fno_${SEED}/best_model.pth"
  local g="$CK/${pre}sg_dt_fno_${SEED}/best_model.pth"
  local miss=0
  for f in "$a" "$d" "$g"; do
    [ -f "$f" ] || { echo "  [!] missing $f"; miss=1; }
  done
  if [ "$miss" = 1 ]; then
    echo "  [!] skipping $label -- set its prefix explicitly, e.g."
    echo "        LH2_PREFIX=... COMB_PREFIX= CYL_PREFIX=... bash $0"
    available
    return 0
  fi
  echo ">>> $label  ($cfg, h=$H, anchor=$anchor)"
  python scripts/plot_fields.py --config "$cfg" \
    --checkpoints ar_fno="$a" dt_fno="$d" sg_dt_fno="$g" \
    --h "$H" --anchor "$anchor" --label "$label" --tag "$tag" \
    --out "$OUT"
}

run_fields configs/lifted_h2.yaml "$LH2_PREFIX"  "B1 Lifted H2"    lh2      "${ANCHOR_LH2:-0}"
run_fields configs/base.yaml      "$COMB_PREFIX" "B2 RealPDEBench" comb     "${ANCHOR_COMB:-0}"
run_fields configs/cylinder.yaml  "$CYL_PREFIX"  "A2 Cylinder"     cylinder "${ANCHOR_CYL:-0}"

if [ -n "$PAPER_FIGS" ]; then
  echo
  echo ">>> installing into $PAPER_FIGS"
  mkdir -p "$PAPER_FIGS"
  for pair in "lh2:fig4_fields_h128_lh2" \
              "comb:fig4_fields_h128_comb" \
              "cylinder:fig4_fields_cylinder_h128"; do
    src="$OUT/fig4_fields_${pair%%:*}_h${H}.pdf"
    dst="$PAPER_FIGS/${pair##*:}.pdf"
    if [ -f "$src" ]; then cp "$src" "$dst"; echo "    $(basename "$dst")"
    else echo "    [!] missing: $src"; fi
  done
fi

echo
echo "================================================================"
echo "CAPTION MATERIAL"
echo "================================================================"
shopt -s nullglob
for f in "$OUT"/fig4_fields_*_h${H}.txt; do
  echo
  cat "$f"
done
cat <<'MSG'
----------------------------------------------------------------
Each caption must now state the trajectory, anchor and horizon, because the
title that used to carry them is gone. The .txt files above have them.

Two things not to copy between captions:
  * the reaction-zone sentence belongs only to the two combustion figures;
    Cylinder has no contour and saying it does would be a false statement
    about the image;
  * the row list, which differs per dataset.

The colour scale of each row is fixed to the GROUND TRUTH's range, not to the
joint range across models. A diverged autoregressive panel therefore saturates
rather than rescaling every other panel into invisibility -- worth one clause
in the caption, since a reader may otherwise read the saturation as clipping.
MSG
