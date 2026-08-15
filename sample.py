"""
Builds the fitted dataset from the staged 1-second grids.

One sample definition shared by every script, so a difference between two
results is never a difference in how the data was built. Three filters, all
mandatory: a usable two-sided quote, 90% coverage of the scored seconds, and a
window opening on or after the fee change.
"""
from __future__ import annotations

import glob
import pathlib

import numpy as np
import pandas as pd

import config

OBS_COLS = ["market_id", "seconds_to_close", "mid_price", "best_bid",
            "best_ask", "book_valid"]

MARKET_COLS = ["condition_id", "market_start_utc", "market_end_utc",
               "winning_outcome", "window_seconds", "order_price_min_tick_size"]


def logit(p, clip=config.MID_CLIP):
    """Map a probability onto the real line, clipped away from the endpoints."""
    p = np.clip(p, clip, 1 - clip)
    return np.log(p / (1 - p))


def load_quotes(stage_dir: pathlib.Path, verbose: bool = True) -> pd.DataFrame:
    """Staged grids, filtered to quote-seconds with a usable two-sided book."""
    files = sorted(pathlib.Path(stage_dir).glob("obsx_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no obsx_*.parquet under {stage_dir}")
    if verbose:
        print(f"[sample] reading {len(files)} staged files", flush=True)

    parts = []
    for i, f in enumerate(files):
        df = pd.read_parquet(f, columns=OBS_COLS)
        # book_valid is inherited from upstream, so the range and ordering
        # conditions are re-asserted rather than trusted. A zero bid is kept on
        # purpose: dropping it removes 4.3% of quotes at a median midpoint of
        # 0.005, precisely when the market has resolved against that side.
        df = df[df.book_valid
                & np.isfinite(df.mid_price)
                & np.isfinite(df.best_bid) & np.isfinite(df.best_ask)
                & (df.best_ask > df.best_bid)
                & (df.best_ask > 0) & (df.best_ask <= 1)
                & (df.best_bid >= 0) & (df.best_bid < 1)]
        parts.append(df.drop(columns=["book_valid"]))
        if verbose and (i + 1) % 600 == 0:
            print(f"  {i + 1}/{len(files)}", flush=True)

    obs = pd.concat(parts, ignore_index=True)
    # staged files overlap at hour boundaries
    return obs.drop_duplicates(["market_id", "seconds_to_close"])


def load_outcomes(markets_glob: str) -> pd.DataFrame:
    """Settlement outcome and close time per market, for the 5-minute series."""
    files = sorted(glob.glob(str(markets_glob)))
    if not files:
        raise FileNotFoundError(f"no market metadata matching {markets_glob}")
    m = pd.concat((pd.read_parquet(f, columns=MARKET_COLS) for f in files),
                  ignore_index=True)
    m = m.drop_duplicates("condition_id", keep="first")
    m = m[m.window_seconds == config.HORIZON_S].copy()
    m["close_utc"] = pd.to_datetime(m.market_end_utc, utc=True)
    # Boundaries apply to the window's open, which the metadata carries directly.
    m["open_utc"] = pd.to_datetime(m.market_start_utc, utc=True)
    m["up"] = m.winning_outcome.map({"Up": 1, "Down": 0})
    m["tick"] = m.order_price_min_tick_size.astype(float)   # varies per market
    return m[m.up.notna()][["condition_id", "open_utc", "close_utc", "up", "tick"]]


def coverage_gate(obs: pd.DataFrame) -> set:
    """Markets whose scored seconds are at least 90% populated.

    Counts seconds 1..300 only. seconds_to_close spans 0..300, which is 301
    values, and second 0 belongs to no bucket, so including it would let a full
    window score 301/300 and turn a 90% gate into an 89.7% one.
    """
    scored = obs[(obs.seconds_to_close >= 1)
                 & (obs.seconds_to_close <= config.HORIZON_S)]
    cov = (scored.groupby("market_id").seconds_to_close.nunique()
           / float(config.HORIZON_S))
    return set(cov[cov >= config.MIN_COVERAGE].index)


def assign_buckets(seconds_to_close: np.ndarray) -> np.ndarray:
    """Time-to-close bucket per second, or -1 if outside them.

    Edges are (lower, upper], so each second lands in exactly one bucket and
    second 0, the settlement instant, lands in none.
    """
    lab = np.full(len(seconds_to_close), -1)
    for i, (hi, lo) in enumerate(config.BUCKETS):
        lab[(seconds_to_close > lo) & (seconds_to_close <= hi)] = i
    return lab


def build(verbose: bool = True) -> pd.DataFrame:
    """One row per (market, second) with outcome, bucket and split."""
    p = config.paths()
    obs = load_quotes(p["stage"], verbose=verbose)
    outcomes = load_outcomes(p["markets_glob"])
    good = coverage_gate(obs)

    d = obs.merge(outcomes, left_on="market_id", right_on="condition_id",
                  how="inner")
    d = d[d.market_id.isin(good)].copy()
    d = d[d.open_utc >= config.REGIME_CUT].copy()

    d["bucket"] = assign_buckets(d.seconds_to_close.to_numpy())
    d = d[d.bucket >= 0]
    d["split"] = np.where(d.open_utc < config.OOS_CUT, "train", "oos")
    d["day"] = d.open_utc.dt.floor("D")
    d["half_spread_c"] = (d.best_ask - d.best_bid) * 100.0 / 2.0
    d["subcent"] = ~np.isclose(d.best_ask * 100, (d.best_ask * 100).round(),
                               atol=1e-6)

    if verbose:
        print(f"[sample] {len(d):,} quote-seconds, "
              f"{d.market_id.nunique():,} windows, "
              f"opens {d.open_utc.min().date()} to {d.open_utc.max().date()}",
              flush=True)
    return d


def bucket_label(i: int) -> str:
    hi, lo = config.BUCKETS[i]
    return f"{hi}-{lo}s"
