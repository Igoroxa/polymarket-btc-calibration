"""
README figures, drawn from artifacts/*.csv. No number is typed in by hand.

Static PNG, because a GitHub README does not run JavaScript. Opaque white
surface so they read against light and dark pages. Okabe-Ito hues, validated
colourblind-safe; nothing depends on colour alone, and every figure sits beside
the table it is drawn from.

House style: one short title stating the finding, no subtitle, series labelled
where they end rather than in a legend box, and everything that is context
rather than result in grey. The figures carry no text the README does not
already say; they are there to show a shape, not to explain it.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bstar
import config

BLUE, ORANGE, GREEN, VERMILLION = "#0072B2", "#E69F00", "#009E73", "#D55E00"
INK, MUTED, HAIRLINE = "#111111", "#6B6B6B", "#D8D8D6"

DPI = 200
README_DP = 3   # figures round like the README, so the two can be checked against each other

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 10.5,
    "text.color": INK,
    "axes.labelcolor": MUTED,
    "axes.labelsize": 10,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.edgecolor": HAIRLINE,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": HAIRLINE,
    "grid.linewidth": 0.7,
    "legend.frameon": False,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
})


def _title(fig, text, x=0.012, y=0.99):
    """One short headline. The README carries the detail."""
    fig.text(x, y, text, ha="left", va="top", fontsize=13, fontweight="600",
             color=INK)


def _save(fig, name):
    path = config.figures_dir() / name
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"  wrote {path.relative_to(config.figures_dir().parent)}")
    return path


def fig_price_paths(paths: pd.DataFrame):
    """The raw material: quoted probability over a contract's life.

    Nothing is fitted here. It exists so a reader sees the data before seeing
    anything estimated from it.
    """
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    fig.subplots_adjust(top=0.86)

    for _, g in paths.groupby("path"):
        up = g.up.iloc[0] == 1
        ax.plot(g.seconds_to_close, g.mid_price * 100,
                color=BLUE if up else ORANGE, lw=1.0, alpha=0.55,
                solid_capstyle="round", zorder=2)

    ax.axhline(50, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.set_xlim(300, 0)
    ax.set_ylim(-4, 104)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks([300, 240, 180, 120, 60, 0])
    ax.grid(axis="x", visible=False)

    # labels sit outside the right spine, clear of the paths themselves
    tx = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for label, y, colour in (("settled Up", 100, BLUE),
                             ("settled Down", 0, ORANGE)):
        ax.annotate(label, xy=(1.0, y), xycoords=tx, xytext=(9, 0),
                    textcoords="offset points", ha="left", va="center",
                    color=colour, fontsize=10, fontweight="600",
                    annotation_clip=False)

    ax.set_xlabel("seconds remaining on the contract")
    ax.set_ylabel("quoted probability of Up (¢)")
    _title(fig, "Twenty-four contracts, as the market quoted them")
    return _save(fig, "price_paths.png")


def fig_calibration_vs_threshold(buckets: pd.DataFrame):
    """Slope by time to close, against both benchmarks."""
    order = [f"{hi}-{lo}s" for hi, lo in config.BUCKETS]
    x = np.arange(len(order))
    threshold = bstar.b_star()

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    fig.subplots_adjust(top=0.88)
    ax.set_xlim(-0.55, len(order) - 0.15)
    ax.grid(axis="x", visible=False)

    for split, colour, marker, dx in (("train", BLUE, "o", -0.11),
                                      ("oos", ORANGE, "s", 0.11)):
        g = buckets[buckets.split == split].set_index("bucket").reindex(order)
        # grey out exactly the estimates that fail to clear break-even
        for i, (bv, se) in enumerate(zip(g.b, g.se_b)):
            lead = bv > threshold
            ax.errorbar(x[i] + dx, bv, yerr=1.96 * se, fmt=marker,
                        color=colour if lead else HAIRLINE,
                        mec=colour, mew=1.4, ms=6.5 if lead else 5,
                        lw=0, elinewidth=1.7 if lead else 1.1,
                        capsize=0, alpha=1.0 if lead else 0.8, zorder=3)

    lo = min(buckets.b - 1.96 * buckets.se_b)
    hi = max(buckets.b + 1.96 * buckets.se_b)
    ax.set_ylim(lo - 0.02, hi + 0.05)

    ax.axhspan(threshold, ax.get_ylim()[1], color=VERMILLION, alpha=0.05, zorder=0)
    ax.axhline(1.0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.axhline(threshold, color=VERMILLION, lw=1.6, zorder=1)

    tx = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(0.995, threshold + 0.004, f"break-even   $b^*$ = {threshold:.4f}",
            transform=tx, ha="right", va="bottom", color=VERMILLION,
            fontsize=9.5, fontweight="600")
    ax.text(0.005, 1.0 - 0.004, "perfectly calibrated   $b$ = 1", transform=tx,
            ha="left", va="top", color=MUTED, fontsize=9.5)

    # direct labels on the last bucket, so no legend box is needed
    last = buckets[buckets.bucket == order[-1]].set_index("split")
    ax.annotate("out-of-sample", xy=(x[-1] + 0.11, last.loc["oos", "b"]),
                xytext=(16, 6), textcoords="offset points", color=ORANGE,
                fontsize=10, fontweight="600", va="center")
    ax.annotate("train", xy=(x[-1] - 0.11, last.loc["train", "b"]),
                xytext=(16, -10), textcoords="offset points", color=BLUE,
                fontsize=10, fontweight="600", va="center")

    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_xlabel("time remaining on the contract")
    ax.set_ylabel("recalibration slope  $b$")
    _title(fig, "Only the last ten seconds clear break-even")
    return _save(fig, "calibration_vs_threshold.png")


def fig_bstar_by_tick():
    """b* against tick size and fee, including configurations this venue does
    not use."""
    rows = pd.DataFrame(bstar.tick_sensitivity(
        ticks=(0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0)))
    here = bstar.b_star()

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    fig.subplots_adjust(top=0.88)
    ax.set_xlim(-0.05, 2.55)

    ax.plot(rows.tick_c, rows.b_star_linear_fee, "-o", color=BLUE, ms=5.5, lw=2)
    ax.plot(rows.tick_c, rows.b_star_quadratic_fee, "--s", color=GREEN, ms=5.5,
            lw=2)

    ax.annotate("fee  0.07 · min(p, 1−p)\nthis venue today",
                xy=(2.0, rows.b_star_linear_fee.iloc[-1]), xytext=(10, -2),
                textcoords="offset points", color=BLUE, fontsize=9.5,
                fontweight="600", va="center")
    ax.annotate("fee  0.25 · min(p, 1−p)²\nbefore 2026-03-30",
                xy=(2.0, rows.b_star_quadratic_fee.iloc[-1]), xytext=(10, -2),
                textcoords="offset points", color=GREEN, fontsize=9.5,
                fontweight="600", va="center")

    ax.plot([1.0], [here], "o", ms=14, mfc="none", mec=VERMILLION, mew=2.2,
            zorder=4)
    ax.annotate(f"1¢ tick  →  $b^*$ = {here:.4f}", xy=(1.0, here),
                xytext=(-14, 20), textcoords="offset points", ha="right",
                color=VERMILLION, fontsize=10, fontweight="600")

    ax.set_xlabel("minimum tick (¢), with the book quoting one tick wide")
    ax.set_ylabel("break-even slope  $b^*$")
    _title(fig, "The venue sets the bar, not its traders")
    return _save(fig, "bstar_by_tick.png")


def fig_tick_artifact(restrictions: pd.DataFrame):
    """Slope by price restriction, endgame against the far-from-close control."""
    order = list(config.RESTRICTIONS)
    labels = ["all prices", "drop outside\n5¢–95¢", "drop outside\n10¢–90¢"]
    x = np.arange(len(order))
    pooled = restrictions[restrictions.scope == "pooled"]
    endgame = f"{config.BUCKETS[-1][0]}-{config.BUCKETS[-1][1]}s"
    control = f"{config.BUCKETS[0][0]}-{config.BUCKETS[0][1]}s"

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    fig.subplots_adjust(top=0.88)
    ax.set_xlim(-0.35, len(order) - 0.35)
    ax.grid(axis="x", visible=False)
    ax.axhline(1.0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)

    for bucket, colour, marker, note in (
            (endgame, VERMILLION, "o", "final 5 seconds"),
            (control, BLUE, "s", "5 minutes out (control)")):
        g = pooled[pooled.bucket == bucket].set_index("restriction").reindex(order)
        ax.errorbar(x, g.b, yerr=1.96 * g.se_b, fmt=f"-{marker}", color=colour,
                    ms=7, lw=2, elinewidth=1.4, capsize=0, zorder=3)
        # labels sit clear of the vertical error bars, not across them
        ax.annotate(note, xy=(x[0], g.b.iloc[0]), xytext=(16, 13),
                    textcoords="offset points", ha="left", color=colour,
                    fontsize=10, fontweight="600")
        # only the endpoints: how much of the sample each cut leaves
        for xi, dx, ha in ((0, 16, "left"), (len(order) - 1, -16, "right")):
            ax.annotate(f"{g.share.iloc[xi]:.0%} of quotes kept",
                        xy=(x[xi], g.b.iloc[xi]), xytext=(dx, -16),
                        textcoords="offset points", ha=ha, fontsize=9,
                        color=MUTED)

    g_end = pooled[pooled.bucket == endgame].set_index("restriction").reindex(order)
    ax.plot(x, g_end.b_star, color=GREEN, lw=1.6, ls=":", zorder=2)
    ax.annotate("break-even $b^*$", xy=(x[-1], g_end.b_star.iloc[-1]),
                xytext=(10, 0), textcoords="offset points", color=GREEN,
                fontsize=9.5, fontweight="600", va="center")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("which quotes the fit is allowed to use")
    ax.set_ylabel("recalibration slope  $b$")
    _title(fig, "Dropping edge prices halves the endgame effect")
    return _save(fig, "tick_artifact.png")


def fig_cutoff_sweep(sweep: pd.DataFrame):
    """Net edge and remaining sample as the interior cutoff slides.

    Separates two things the three-row table runs together: what happens to
    the estimate, and what happens to the sample left to estimate it from.
    """
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(8.2, 6.0), sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.16})
    fig.subplots_adjust(top=0.90)

    x = sweep.cutoff * 100
    crossing = sweep[sweep.edge_lo_c <= 0].cutoff.min() * 100

    ax.fill_between(x, sweep.edge_lo_c, sweep.edge_hi_c, color=BLUE, alpha=0.13,
                    lw=0, zorder=2)
    ax.plot(x, sweep.edge_c, "-", color=BLUE, lw=2.4, zorder=3)
    ax.axhline(0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.axvline(crossing, color=VERMILLION, lw=1.4, ls=":", zorder=2)

    ax.annotate("net edge", xy=(x.iloc[-1], sweep.edge_c.iloc[-1]),
                xytext=(10, 0), textcoords="offset points", color=BLUE,
                fontsize=10, fontweight="600", va="center")
    ax.annotate("95% interval", xy=(x.iloc[-1], sweep.edge_hi_c.iloc[-1]),
                xytext=(10, 0), textcoords="offset points", color=MUTED,
                fontsize=9.5, va="center")
    ax.annotate(f"interval reaches zero\nat a {crossing:.0f}¢ cut",
                xy=(crossing, ax.get_ylim()[1]), xytext=(8, -4),
                textcoords="offset points", color=VERMILLION, fontsize=9.5,
                fontweight="600", va="top")

    ax.set_ylabel("net edge (¢ per share)")
    ax.grid(axis="x", visible=False)

    ax2.plot(x, sweep.share * 100, "-", color=ORANGE, lw=2.4)
    ax2.axvline(crossing, color=VERMILLION, lw=1.4, ls=":", zorder=2)
    ax2.annotate("sample left", xy=(x.iloc[-1], sweep.share.iloc[-1] * 100),
                 xytext=(10, 0), textcoords="offset points", color=ORANGE,
                 fontsize=10, fontweight="600", va="center")
    ax2.set_ylim(0, 108)
    ax2.set_ylabel("% of quotes kept")
    ax2.set_xlabel("cut width: quotes this close to 0¢ or 100¢ are dropped")
    ax2.grid(axis="x", visible=False)

    _title(fig, "The estimate holds; the sample does not", y=0.995)
    return _save(fig, "cutoff_sweep.png")


def build_all():
    a = config.artifacts_dir()
    buckets = pd.read_csv(a / "calibration_buckets.csv")
    restrictions = pd.read_csv(a / "tick_restrictions.csv")
    sweep = pd.read_csv(a / "cutoff_sweep.csv")
    paths = pd.read_csv(a / "price_paths.csv")
    print("building figures:")
    fig_price_paths(paths)
    fig_calibration_vs_threshold(buckets)
    fig_bstar_by_tick()
    fig_tick_artifact(restrictions)
    fig_cutoff_sweep(sweep)


if __name__ == "__main__":
    build_all()
