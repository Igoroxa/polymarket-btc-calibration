"""
What a miscalibration is worth after costs, and the slope at which it is worth
nothing. Published exchange parameters only; nothing here is fitted.

    net edge = 100*(q - p) - half_spread - 100*r*min(p, 1-p)**e
    where     q = sigma(a + b*logit(p))

`net_edge` answers the economic question and `b_star` restates it as a
threshold. See README Part 3.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

TICK_C = 1.0
FEE_R = 0.07
FEE_E = 1
HALF_SPREAD_C = TICK_C / 2.0

# TICK_C is the grid the book quotes on. The exchange's stated minimum is finer
# on roughly 40% of markets, but the median half-spread is 0.500c in both
# groups, so 1c is what binds. The fee applies from 2026-03-30; before that it
# was 0.25*x**2.

_GRID = 2_000_001


def _surface(a, b, half_c, r, e, lo, hi, n):
    p = np.linspace(lo, hi, n)
    q = 1.0 / (1.0 + np.exp(-(a + b * np.log(p / (1.0 - p)))))
    return p, q, 100.0 * (q - p) - half_c - 100.0 * r * np.minimum(p, 1.0 - p) ** e


def net_edge(a, b, half_c=HALF_SPREAD_C, r=FEE_R, e=FEE_E,
             lo=1e-6, hi=1 - 1e-6, n=_GRID):
    """Best net edge anywhere on the price surface, cents per share.

    A ceiling: it assumes the fit is right and that the trade happens at the
    single best price, instantly.
    """
    return float(np.max(_surface(a, b, half_c, r, e, lo, hi, n)[2]))


def best_price(a, b, half_c=HALF_SPREAD_C, r=FEE_R, e=FEE_E,
               lo=1e-6, hi=1 - 1e-6, n=_GRID):
    """The price at which net_edge is attained."""
    p, _, edge = _surface(a, b, half_c, r, e, lo, hi, n)
    return float(p[int(np.argmax(edge))])


def net_edge_se(a, b, cov, half_c=HALF_SPREAD_C, r=FEE_R, e=FEE_E,
                lo=1e-6, hi=1 - 1e-6):
    """Standard error of net_edge, given the 2x2 covariance of (a, b).

    The off-diagonal matters: a and b come from the same fit and move together.

    Changing a or b also moves the best price, but that can be ignored. At the
    top of a curve the slope is zero, so shifting the best price changes the
    height by nothing to first order. What is left is how the edge responds to
    each coefficient at the best price p*.
    """
    p_star = best_price(a, b, half_c, r, e, lo, hi)
    lg = np.log(p_star / (1 - p_star))
    q = 1.0 / (1.0 + np.exp(-(a + b * lg)))
    grad = np.array([100.0 * q * (1 - q), 100.0 * q * (1 - q) * lg])
    return float(np.sqrt(grad @ np.asarray(cov) @ grad))


def net_edge_ci(a, b, cov, level=1.96, **kw):
    """Net edge with a confidence interval, cents per share."""
    point = net_edge(a, b, **kw)
    se = net_edge_se(a, b, cov, **kw)
    return {"edge_c": point, "se_c": se,
            "lo_c": point - level * se, "hi_c": point + level * se}


def b_star(half_c=HALF_SPREAD_C, r=FEE_R, e=FEE_E, lo=1e-6, hi=1 - 1e-6):
    """Slope at which the best net edge is zero, holding a = 0.

    Fixed at a = 0 because once the level term is free this is a curve rather
    than a number. Restricting [lo, hi] matters when the slope was fitted on
    restricted prices: one estimated on 10c-90c quotes cannot be traded at 92c.
    """
    return float(brentq(lambda b: net_edge(0.0, b, half_c, r, e, lo, hi),
                        1.0, 3.0))


def tick_sensitivity(ticks=(0.1, 0.25, 0.5, 1.0, 2.0)):
    """b* under other tick sizes and fee schedules, book quoting one tick wide."""
    return [{"tick_c": t,
             "half_spread_c": t / 2.0,
             "b_star_linear_fee": b_star(t / 2.0, 0.07, 1),
             "b_star_quadratic_fee": b_star(t / 2.0, 0.25, 2)}
            for t in ticks]


# quoted in the README; tests and verify.py read them from here
PUBLISHED = {
    "all_prices": 1.0629,
    "interior_10_90": 1.0642,
    "old_regime": 1.0072,
}


if __name__ == "__main__":
    for label, args, expected in (
        ("this venue, all prices", (0.5, 0.07, 1), PUBLISHED["all_prices"]),
        ("this venue, 10c-90c", (0.5, 0.07, 1, 0.10, 0.90),
         PUBLISHED["interior_10_90"]),
        ("0.1c tick, quadratic fee", (0.05, 0.25, 2), PUBLISHED["old_regime"]),
    ):
        got = b_star(*args)
        status = "ok" if round(got, 4) == expected else "MISMATCH"
        print(f"{label:26s} b* = {got:.4f}  expected {expected}  [{status}]")
