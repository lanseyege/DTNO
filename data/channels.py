"""
Channel semantics.

Two jobs, both of which exist because §26 of the proposal insists that a single
averaged RMSE is not an acceptable headline number:

1. `channel_group` maps a raw channel name onto one of
   {flow, thermo, chemistry, reaction} so per-variable errors can be reported
   separately.  A model that nails velocity and misses the reaction zone must
   not be able to hide behind the mean.

2. `default_transform` guesses a preprocessing transform from the name
   (species -> log, heat release -> symlog, everything else -> plain z-score).
   The guess is only a default: `scripts/audit_data.py` measures the actual
   dynamic range and writes an explicit per-channel decision into
   `norm_stats.json`, and the config can override any of it.

Name matching is deliberately loose (case-insensitive substring / regex) because
RealPDEBench, REALM and BLASTNet all spell the same physics differently
(`Mole_Fraction_of_OH` vs `Y_OH` vs `OH`).  Run the audit first; it prints the
names it actually found and the group it assigned to each.
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence

GROUPS = ("flow", "thermo", "chemistry", "reaction")

# BLASTNet spells velocity UX / UY / UZ and mass fractions YH2O / YOH, neither
# of which matched the RealPDEBench-derived patterns: "UX" is not "^u$", and
# "\bh2o\b" does not fire inside "YH2O" because Y is a word character. The
# result was every Lifted-H2 channel landing in `thermo`, which silently
# disables the per-variable reporting §26 exists to guarantee. Always check the
# grouping the audit prints when adding a dataset.
_FLOW = re.compile(
    r"velocity|vel\[|^u$|^v$|^w$|_u$|_v$|_w$|^u[xyz]$|^vel[xyz]$|"
    r"vorticity|momentum|mach", re.I)
#_PRESSURE = re.compile(r"pressure", re.I)
#_THERMO = re.compile(r"temperature|^t$|density|rho|enthalpy|entropy", re.I)
_PRESSURE = re.compile(r"pressure|^p$|_p$|\bpres\b", re.I)
_THERMO   = re.compile(r"temperature|^t$|density|rho|enthalpy|entropy|buoyancy", re.I)
_REACTION = re.compile(
    r"heat[_ ]?release|hrr|reaction[_ ]?rate|source[_ ]?term|q_?dot|qdot", re.I)
_SPECIES = re.compile(
    r"mole[_ ]?fraction|mass[_ ]?fraction|^y_|_y_|concentration|"
    # BLASTNet style: YH2O, YOH, YCH4 -- a leading Y glued to a formula
    r"^y(h2o|oh|h2|o2|n2|co2|co|ch4|nh3|nh2|no|c2h4|ch2o)$|"
    r"\b(ch4|co2|co|h2o|h2|o2|oh|nh3|nh2|no|n2|c2h4|ch2o|h|o|n)\b", re.I)

# Species most people actually plot for an NH3/CH4 flame, in a sensible order.
CANONICAL_SPECIES = ["OH", "CH4", "NH3", "NH2", "CO", "CO2", "H2O", "H2", "O2", "N2"]


def channel_group(name: str) -> str:
    """flow | thermo | chemistry | reaction — first match wins, order matters."""
    if _REACTION.search(name):
        return "reaction"
    if _FLOW.search(name):
        return "flow"
    if _PRESSURE.search(name):
        return "flow"          # pressure travels with the hydrodynamics (§26)
    if _THERMO.search(name):
        return "thermo"
    if _SPECIES.search(name):
        return "chemistry"
    return "thermo"            # conservative fallback: reported, never hidden


def group_channels(names: Sequence[str]) -> Dict[str, List[int]]:
    """{'flow': [0,3,4], 'chemistry': [...], ...} — empty groups are dropped."""
    out: Dict[str, List[int]] = {g: [] for g in GROUPS}
    for i, n in enumerate(names):
        out[channel_group(n)].append(i)
    return {g: idx for g, idx in out.items() if idx}


def default_transform(name: str) -> str:
    """'zscore' | 'log' | 'symlog' — see data/transforms.py for the maths."""
    if _REACTION.search(name):
        return "symlog"        # heat release can be signed (endothermic zones)
    if _SPECIES.search(name) and not _FLOW.search(name):
        return "log"           # spans many decades, strictly non-negative
    return "zscore"


# ---------------------------------------------------------------------------
# Named lookups used by the combustion metrics
# ---------------------------------------------------------------------------

def find_channel(names: Sequence[str], *patterns: str) -> int:
    """Index of the first channel matching any pattern, else -1.

    >>> find_channel(names, "heat_release", "hrr")
    """
    for pat in patterns:
        rx = re.compile(pat, re.I)
        for i, n in enumerate(names):
            if rx.search(n):
                return i
    return -1


def reaction_zone_channel(names: Sequence[str]) -> int:
    """Channel used to define the flame mask (§29): heat release, else OH."""
    i = find_channel(names, r"heat[_ ]?release", r"hrr", r"q_?dot")
    if i >= 0:
        return i
    # No heat-release channel (BLASTNet Lifted H2 ships none), so fall back to
    # OH, which marks the reaction layer of a hydrogen flame well enough for the
    # §29 mask. Note this in any figure caption: the mask is an OH iso-contour,
    # not a heat-release one, and the two are not interchangeable across
    # datasets.
    return find_channel(names, r"\bOH\b", r"_OH$", r"of_OH", r"^yoh$")


def temperature_channel(names: Sequence[str]) -> int:
    # "^T$" must be matched case-SENSITIVELY-ish: find_channel is re.I, so a
    # bare "T" is checked last, after the explicit spellings.
    return find_channel(names, r"^temperature$", r"temperature", r"^t_k$", r"^T$")


def velocity_channels(names: Sequence[str]) -> List[int]:
    """In-plane velocity components, in (u, v) order where recoverable."""
    idx = []
    for pat in (r"velocity\[i\]", r"^ux$", r"\bu\b", r"velocity_x", r"x_velocity"):
        i = find_channel(names, pat)
        if i >= 0:
            idx.append(i)
            break
    for pat in (r"velocity\[j\]", r"^uy$", r"\bv\b", r"velocity_y", r"y_velocity"):
        i = find_channel(names, pat)
        if i >= 0 and i not in idx:
            idx.append(i)
            break
    return idx
