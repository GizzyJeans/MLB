"""Removal of bookmaker margin ("vig") from two-way market prices.

Every estimator here takes *decimal* odds and returns fair probabilities that
sum to one. Four are provided rather than one because they agree almost
exactly on balanced prices (-110/-110) and diverge on lopsided ones. That
divergence is the useful part: it is a direct measure of how much of a
"fair probability" is assumption rather than observation, and it belongs in
any honest edge calculation.

Reference points for the two-way case:

  multiplicative  splits margin in proportion to implied probability. Biases
                  the favourite's fair price upward relative to Shin.
  additive        splits margin equally in absolute terms. Can emit negative
                  probabilities on extreme longshots.
  power           solves sum(q_i**k) == 1. Between multiplicative and Shin.
  shin            models margin as protection against insider order flow,
                  which loads more of the vig onto the longshot. Closest to
                  observed closing-line behaviour in most studies.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

__all__ = [
    "american_to_decimal",
    "decimal_to_american",
    "implied_probabilities",
    "overround",
    "devig",
    "devig_multiplicative",
    "devig_additive",
    "devig_power",
    "devig_shin",
    "METHODS",
]


def american_to_decimal(american: float) -> float:
    """Convert American odds to decimal odds (stake included)."""
    if american > 0:
        return 1.0 + american / 100.0
    if american < 0:
        return 1.0 + 100.0 / abs(american)
    raise ValueError("American odds of 0 are undefined")


def decimal_to_american(decimal: float) -> float:
    """Convert decimal odds to American odds."""
    if decimal <= 1.0:
        raise ValueError(f"decimal odds must exceed 1.0, got {decimal}")
    profit = decimal - 1.0
    if profit >= 1.0:
        return 100.0 * profit
    return -100.0 / profit


def probability_to_american(probability: float) -> float:
    """Fair American odds for a probability, ignoring margin."""
    if not 0.0 < probability < 1.0:
        raise ValueError(f"probability must be in (0, 1), got {probability}")
    return decimal_to_american(1.0 / probability)


def implied_probabilities(decimals: Sequence[float]) -> list[float]:
    """Raw (margin-inclusive) implied probabilities."""
    return [1.0 / d for d in decimals]


def overround(decimals: Sequence[float]) -> float:
    """Bookmaker margin as a fraction, e.g. 0.045 for a 4.5% hold."""
    return sum(implied_probabilities(decimals)) - 1.0


def devig_multiplicative(decimals: Sequence[float]) -> list[float]:
    """Scale implied probabilities down proportionally."""
    q = implied_probabilities(decimals)
    total = sum(q)
    return [x / total for x in q]


def devig_additive(decimals: Sequence[float]) -> list[float]:
    """Subtract the margin equally from each outcome.

    Falls back to the multiplicative result when equal subtraction would push
    an outcome to or below zero, which happens on heavy longshots.
    """
    q = implied_probabilities(decimals)
    excess = (sum(q) - 1.0) / len(q)
    fair = [x - excess for x in q]
    if any(p <= 0.0 for p in fair):
        return devig_multiplicative(decimals)
    return fair


def _bisect(f: Callable[[float], float], lo: float, hi: float,
            tol: float = 1e-12, max_iter: int = 200) -> float:
    """Root-find on a monotone function, returning the midpoint on failure."""
    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0:
        return lo if abs(f_lo) < abs(f_hi) else hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) < tol or (hi - lo) < tol:
            return mid
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def devig_power(decimals: Sequence[float]) -> list[float]:
    """Solve for the exponent k with sum(q_i ** k) == 1."""
    q = implied_probabilities(decimals)
    k = _bisect(lambda kk: sum(x ** kk for x in q) - 1.0, 1.0, 20.0)
    fair = [x ** k for x in q]
    total = sum(fair)
    return [p / total for p in fair]


def devig_shin(decimals: Sequence[float]) -> list[float]:
    """Shin (1993) margin model.

    Solves for the insider-trading proportion z that makes the recovered
    probabilities sum to one, then inverts the Shin price relation.
    """
    q = implied_probabilities(decimals)
    total = sum(q)

    def shin_probs(z: float) -> list[float]:
        if z <= 0.0:
            return devig_multiplicative(decimals)
        out = []
        for x in q:
            root = math.sqrt(z * z + 4.0 * (1.0 - z) * (x * x) / total)
            out.append((root - z) / (2.0 * (1.0 - z)))
        return out

    z = _bisect(lambda zz: sum(shin_probs(zz)) - 1.0, 0.0, 0.9)
    fair = shin_probs(z)
    norm = sum(fair)
    return [p / norm for p in fair]


METHODS: dict[str, Callable[[Sequence[float]], list[float]]] = {
    "multiplicative": devig_multiplicative,
    "additive": devig_additive,
    "power": devig_power,
    "shin": devig_shin,
}


def devig(decimals: Sequence[float], method: str = "shin") -> list[float]:
    """De-vig with the named method."""
    try:
        return METHODS[method](decimals)
    except KeyError:
        raise ValueError(
            f"unknown de-vig method {method!r}; choose from {sorted(METHODS)}"
        ) from None


def devig_all(decimals: Sequence[float]) -> dict[str, list[float]]:
    """Run every estimator, for measuring method risk."""
    return {name: fn(decimals) for name, fn in METHODS.items()}


def method_spread(decimals: Sequence[float], index: int = 0) -> float:
    """Range of fair probabilities across estimators for one outcome.

    Treat this as a floor on the uncertainty of any single de-vigged number.
    An edge smaller than this spread is not distinguishable from a choice of
    de-vig convention.
    """
    values = [probs[index] for probs in devig_all(decimals).values()]
    return max(values) - min(values)
