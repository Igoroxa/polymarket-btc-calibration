"""
The recalibration fit: P(Up | mid = p) = sigma(a + b*logit(p)).

a = 0 and b = 1 is a perfectly calibrated market. b > 1 means the truth sits
further from 50% than the quote. Both coefficients are carried forward with
their covariance, since the net edge depends on both.

Standard errors are grouped, and this is required rather than a refinement. A
window has one outcome shared by all ~300 of its seconds, so those rows carry
one observation's worth of information about it, not 300.

Produces the bucket table, the price-restriction table, the cutoff sweep and
the per-market tick comparison. See README Parts 2 to 4.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

import bstar
import config
import sample


def fit(y, x, groups) -> dict:
    """Logistic regression of y on x, standard errors grouped by `groups`.

    Returns the full 2x2 covariance, not just the diagonal: the net edge needs
    to know how a and b move together.
    """
    X = sm.add_constant(x)
    res = sm.GLM(y, X, family=sm.families.Binomial()).fit(
        maxiter=200, cov_type="cluster", cov_kwds={"groups": groups})
    a, b = res.params
    V = np.asarray(res.cov_params())
    se_a, se_b = np.sqrt(np.diag(V))
    return {"a": float(a), "se_a": float(se_a),
            "b": float(b), "se_b": float(se_b),
            "cov_ab": float(V[0, 1]),
            "converged": bool(res.converged)}


def _fit_frame(g: pd.DataFrame, cluster: str = "market_id") -> dict:
    return fit(g.up.to_numpy(),
               sample.logit(g.mid_price.to_numpy()),
               g[cluster].to_numpy())


def _economics(r: dict) -> dict:
    """Net edge and its interval, from the fitted (a, b) and their covariance."""
    cov = [[r["se_a"] ** 2, r["cov_ab"]], [r["cov_ab"], r["se_b"] ** 2]]
    ci = bstar.net_edge_ci(r["a"], r["b"], cov)
    return {"edge_c": ci["edge_c"], "edge_se_c": ci["se_c"],
            "edge_lo_c": ci["lo_c"], "edge_hi_c": ci["hi_c"],
            "edge_z": ci["edge_c"] / ci["se_c"] if ci["se_c"] > 0 else np.nan,
            "best_price": bstar.best_price(r["a"], r["b"])}


def bucket_table(panel: pd.DataFrame, min_windows: int = 30,
                 cluster: str = "market_id") -> pd.DataFrame:
    """b by time-to-close bucket, fitted separately on each split.

    Reported against both benchmarks. z_vs_1 asks whether the market is
    miscalibrated; z_vs_bstar asks whether that is worth more than it costs to
    correct. The same estimate can be four standard errors from one and within
    one of the other.
    """
    threshold = bstar.b_star()
    rows = []
    for i in range(len(config.BUCKETS)):
        for split in ("train", "oos"):
            g = panel[(panel.bucket == i) & (panel.split == split)]
            if g.market_id.nunique() < min_windows:
                continue
            r = _fit_frame(g, cluster)
            rows.append({
                "bucket": sample.bucket_label(i), "split": split,
                "a": r["a"], "se_a": r["se_a"],
                "b": r["b"], "se_b": r["se_b"], "cov_ab": r["cov_ab"],
                "z_vs_1": (r["b"] - 1.0) / r["se_b"],
                "z_vs_bstar": (r["b"] - threshold) / r["se_b"],
                **_economics(r),
                "n_windows": int(g.market_id.nunique()),
                "n_days": int(g.day.nunique()),
                "n_obs": len(g),
            })
    return pd.DataFrame(rows)


def restriction_table(panel: pd.DataFrame, bucket_idx: int,
                      scopes=("pooled", "train", "oos"),
                      min_windows: int = 100,
                      cluster: str = "market_id") -> pd.DataFrame:
    """b by price restriction within one bucket.

    A 1c grid cannot show 99.4%, so the 99c bucket holds a range of beliefs
    skewed upward and settles Up more often than 99% of the time. Dropping
    boundary prices should shrink b where the grid binds and leave it alone
    where it does not, which the far-from-close bucket checks.

    b* is recomputed per range, since a slope fitted on 10c-90c quotes cannot
    be traded at 92c.
    """
    label = sample.bucket_label(bucket_idx)
    base_all = panel[panel.bucket == bucket_idx]
    rows = []
    for scope in scopes:
        base = base_all if scope == "pooled" else base_all[base_all.split == scope]
        for name, (lo, hi) in config.RESTRICTIONS.items():
            g = base if name == "all" else base[base.mid_price.between(lo, hi)]
            if g.market_id.nunique() < min_windows:
                continue
            r = _fit_frame(g, cluster)
            threshold = bstar.b_star(lo=max(lo, 1e-6), hi=min(hi, 1 - 1e-6))
            rows.append({
                "bucket": label, "scope": scope, "restriction": name,
                "a": r["a"], "se_a": r["se_a"],
                "b": r["b"], "se_b": r["se_b"], "cov_ab": r["cov_ab"],
                "b_star": threshold,
                "z_vs_bstar": (r["b"] - threshold) / r["se_b"],
                **_economics(r),
                "n_windows": int(g.market_id.nunique()),
                "n_obs": len(g),
                "share": len(g) / max(len(base), 1),
            })
    return pd.DataFrame(rows)


def cutoff_sweep(panel: pd.DataFrame, bucket_idx: int,
                 cutoffs=np.arange(0.0, 0.26, 0.01),
                 min_windows: int = 100,
                 cluster: str = "market_id") -> pd.DataFrame:
    """Slide the interior cutoff continuously and refit at each step.

    Three points cannot show whether the result moves as a cliff, a trend or
    noise. This traces it.
    """
    base = panel[panel.bucket == bucket_idx]
    rows = []
    for c in cutoffs:
        lo, hi = float(c), float(1 - c)
        g = base[base.mid_price.between(lo, hi)]
        if g.market_id.nunique() < min_windows:
            continue
        r = _fit_frame(g, cluster)
        threshold = bstar.b_star(lo=max(lo, 1e-6), hi=min(hi, 1 - 1e-6))
        rows.append({
            "cutoff": lo,
            "b": r["b"], "se_b": r["se_b"],
            "a": r["a"], "se_a": r["se_a"], "cov_ab": r["cov_ab"],
            "b_star": threshold,
            **_economics(r),
            "n_windows": int(g.market_id.nunique()),
            "n_obs": len(g),
            "share": len(g) / max(len(base), 1),
        })
    return pd.DataFrame(rows)


def tick_experiment(panel: pd.DataFrame, bucket_idx: int,
                    cluster: str = "market_id") -> pd.DataFrame:
    """Endgame slope by minimum tick: a natural experiment on the artifact.

    The tick is set per market, so if b > 1 is rounding, markets allowed a finer
    grid should show less of it. Reported alongside the quantities that decide
    whether the test could work at all: only ~6% of quotes on fine-tick markets
    use the finer grid, and the median half-spread is 0.500c in both groups.
    """
    label = sample.bucket_label(bucket_idx)
    base = panel[panel.bucket == bucket_idx]
    rows = []
    for tick, g in base.groupby("tick"):
        if g.market_id.nunique() < 100:
            continue
        r = _fit_frame(g, cluster)
        rows.append({
            "bucket": label, "tick": tick,
            "b": r["b"], "se_b": r["se_b"],
            "a": r["a"], "se_a": r["se_a"], "cov_ab": r["cov_ab"],
            **_economics(r),
            "subcent_share": float(g.subcent.mean()),
            "median_half_spread_c": float(g.half_spread_c.median()),
            "n_windows": int(g.market_id.nunique()),
            "n_obs": len(g),
        })
    out = pd.DataFrame(rows)
    if len(out) == 2:
        # disjoint markets, so the two estimates are independent
        coarse = out[out.tick == max(out.tick)].iloc[0]
        fine = out[out.tick == min(out.tick)].iloc[0]
        diff = coarse.b - fine.b
        se = float(np.hypot(coarse.se_b, fine.se_b))
        out.attrs["difference"] = {"b_coarse_minus_fine": diff, "se": se,
                                   "z": diff / se}
    return out


def price_paths(panel: pd.DataFrame, n: int = 24, seed: int = 0,
                every: int = 2) -> pd.DataFrame:
    """A random sample of fitted windows, as the market actually quoted them.

    Illustration only: nothing is fitted to this, and no result depends on it.
    Windows are drawn from the training split and relabelled 1..n, so the
    artifact carries no market identifiers.
    """
    train = panel[panel.split == "train"]
    ids = np.random.default_rng(seed).choice(
        np.sort(train.market_id.unique()), size=n, replace=False)
    order = {m: i + 1 for i, m in enumerate(ids)}
    g = train[train.market_id.isin(ids) & (train.seconds_to_close % every == 0)]
    return (g.assign(path=g.market_id.map(order))
             [["path", "seconds_to_close", "mid_price", "up"]]
             .sort_values(["path", "seconds_to_close"], ascending=[True, False])
             .reset_index(drop=True))


def archive_coverage(panel: pd.DataFrame) -> pd.DataFrame:
    """Windows in the sample against the number the fixed schedule implies.

    The gap is selection upstream of every filter here, and it is larger than
    the coverage gate's own effect.
    """
    per_day = 24 * 60 * 60 // config.HORIZON_S
    rows = []
    for split, start, end in (("train", config.REGIME_CUT, config.OOS_CUT),
                              ("oos", config.OOS_CUT, None)):
        g = panel[panel.split == split]
        if not len(g):
            continue
        last = end or (g.open_utc.max().normalize() + pd.Timedelta(days=1))
        span_days = (last - start).days
        rows.append({
            "split": split,
            "span_days": span_days,
            "days_with_data": int(g.day.nunique()),
            "windows_in_sample": int(g.market_id.nunique()),
            "windows_implied_by_schedule": span_days * per_day,
        })
    out = pd.DataFrame(rows)
    out["captured"] = out.windows_in_sample / out.windows_implied_by_schedule
    return out


def write_artifacts(panel: pd.DataFrame, out_dir=None) -> dict[str, pd.DataFrame]:
    """Fit everything the README reports and write it to artifacts/."""
    out = out_dir or config.artifacts_dir()
    out.mkdir(exist_ok=True)
    endgame_idx, control_idx = len(config.BUCKETS) - 1, 0

    buckets = bucket_table(panel)
    restrictions = pd.concat([
        restriction_table(panel, endgame_idx),
        restriction_table(panel, control_idx, scopes=("pooled",)),
    ], ignore_index=True)
    sweep = cutoff_sweep(panel, endgame_idx)
    coverage = archive_coverage(panel)
    ticks = tick_experiment(panel, endgame_idx)
    # robustness: neighbouring windows grouped together
    restrictions_day = restriction_table(panel, endgame_idx, scopes=("pooled",),
                                         cluster="day")

    tables = {
        "calibration_buckets": buckets,
        "tick_restrictions": restrictions,
        "cutoff_sweep": sweep,
        "archive_coverage": coverage,
        "tick_experiment": ticks,
        "tick_restrictions_day_clustered": restrictions_day,
        "price_paths": price_paths(panel),
    }
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
    return tables


if __name__ == "__main__":
    import sys
    try:
        panel = sample.build()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
    for name, df in write_artifacts(panel).items():
        print(f"\n===== {name} =====")
        print(df.round(4).to_string(index=False))
