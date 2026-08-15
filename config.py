"""
Data paths and the sample boundaries every script shares.

Paths come from config.toml, which is not committed. The constants below are
not configurable: they define what the published results mean, so changing one
silently would change what every table says.
"""
from __future__ import annotations

import pathlib
import tomllib

import pandas as pd

# Fee schedule boundary. The last window on the old schedule opens 2026-03-29,
# the first on the new one opens 2026-03-30, and no day carries both. This is a
# fee boundary only: the minimum tick is set per market and does not change on
# any date.
REGIME_CUT = pd.Timestamp("2026-03-30T00:00:00Z")

OOS_CUT = pd.Timestamp("2026-06-01T00:00:00Z")

MIN_COVERAGE = 0.90

HORIZON_S = 300

# Fractions of the horizon (1.00, 0.66, 0.33) with a fixed fine tail. Set before
# any outcome was examined.
BUCKETS = [(300, 198), (198, 99), (99, 60), (60, 30), (30, 10), (10, 5), (5, 0)]

# Never binds here: the narrowest quote on a 1c grid puts the midpoint at 0.005.
MID_CLIP = 1e-3

RESTRICTIONS = {
    "all": (0.0, 1.0),
    "interior 5-95c": (0.05, 0.95),
    "interior 10-90c": (0.10, 0.90),
}

_ROOT = pathlib.Path(__file__).resolve().parent


def load(path: str | pathlib.Path | None = None) -> dict:
    p = pathlib.Path(path) if path else _ROOT / "config.toml"
    if not p.exists():
        raise FileNotFoundError(
            f"{p.name} not found. Copy config.example.toml to config.toml and "
            "set the paths for your machine. The staged grids are not "
            "redistributed; see the README on rebuilding them."
        )
    with open(p, "rb") as fh:
        return tomllib.load(fh)


def paths(cfg: dict | None = None) -> dict[str, pathlib.Path]:
    d = (cfg or load())["data"]
    return {"stage": pathlib.Path(d["stage_dir"]),
            "markets_glob": d["markets_glob"]}


def artifacts_dir() -> pathlib.Path:
    return _ROOT / "artifacts"


def figures_dir() -> pathlib.Path:
    return _ROOT / "figures"
