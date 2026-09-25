#!/usr/bin/env bash
# Phase 0 (§36) — data audit. Run this first; nothing else will start without it.
set -euo pipefail
CONFIG=${1:-configs/dt_fno.yaml}

python scripts/audit_data.py --config "$CONFIG" --plots

echo
echo "Read the transform table above before continuing."
echo "A species channel left on 'zscore' will dominate the loss;"
echo "override it with data.channel_transforms in the config and re-run --force."
