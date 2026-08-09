"""Pricing for Asian-style handicap and total lines.

These boards quote both sides at a flat price and move the *line* instead,
in a compact notation: an integer, a sign, and a two-digit weight.

    8平    -> 8.00       all stake at 8
    8-30   -> 8.15       30% at 8.5, 70% at 8
    9+70   -> 8.65       70% at 8.5, 30% at 9
    9.5    -> 9.50       all stake at 9.5

The sign says which side of the integer the split sits on: a minus reaches
*up* toward the half above, a plus reaches *down* toward the half below.
Getting that backwards puts a line most of a run away from the market and
manufactures enormous fake edges -- a total decoded as 8.45 when the board
really said 7.55 priced the under at +17%, which is not a number any liquid
market produces. It was the sign, not the market.

A quoted 8.15 is not a number of runs anyone can score. It is a split
ticket, and it must be priced as one: the integer leg can push and return
stake, the half leg cannot. Collapsing it to one number and asking whether
the total cleared 8.15 throws the push away, which is worth real money on
low totals where pushes are common.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

LINE_PATTERN = re.compile(r"^(\d+)(?:([+-])(\d{1,2})|(平)|\.(5))?$")


@dataclass(frozen=True)
class AsianLine:
    """A split ticket: `weight` of the stake at `high`, the rest at `low`."""

    low: float
    high: float
    weight: float
    raw: str

    @property
    def effective(self) -> float:
        return (1.0 - self.weight) * self.low + self.weight * self.high

    def __str__(self) -> str:
        return f"{self.raw}({self.effective:g})"


def parse_line(text: str) -> AsianLine:
    """Decode the board's compact line notation."""
    text = text.strip()
    match = LINE_PATTERN.match(text)
    if not match:
        raise ValueError(f"unrecognised line {text!r}")
    base = float(match.group(1))

    if match.group(4) or match.group(3) is None and match.group(5) is None:
        return AsianLine(base, base, 0.0, text)          # 平 or bare integer
    if match.group(5):
        return AsianLine(base + 0.5, base + 0.5, 0.0, text)   # explicit .5

    weight = int(match.group(3)) / 100.0
    if match.group(2) == "-":
        # Reaches up: `weight` of the stake sits on the half above the base.
        return AsianLine(base, base + 0.5, weight, text)
    # Reaches down: `weight` sits on the half below, so the base is the far leg.
    return AsianLine(base - 0.5, base, 1.0 - weight, text)


def _leg_ev(wins: np.ndarray, pushes: np.ndarray, hk_price: float,
            shift: float) -> float:
    """EV per unit staked on one leg.

    A push returns the stake and contributes nothing, but it also removes the
    outcome from the losing side, which is exactly why integer lines price
    differently from half lines.

    `shift` moves probability from losing to winning, used to correct the
    engine's measured bias rather than pretending it is not there.
    """
    n = len(wins)
    p_win = float(np.sum(wins)) / n + shift
    p_push = float(np.sum(pushes)) / n
    p_win = min(max(p_win, 0.0), 1.0 - p_push)
    p_lose = 1.0 - p_win - p_push
    return p_win * hk_price - p_lose


def handicap_ev(margin: np.ndarray, line: AsianLine, *, hk_price: float,
                laying: bool, shift: float = 0.0) -> float:
    """EV of one unit on an Asian handicap.

    `margin` is the favourite's margin; `laying` is the side giving the runs.
    """
    total = 0.0
    for level, weight in ((line.low, 1.0 - line.weight),
                          (line.high, line.weight)):
        if weight == 0.0:
            continue
        integral = float(level).is_integer()
        if laying:
            wins = margin > level
            pushes = (margin == level) if integral else np.zeros_like(wins)
        else:
            wins = margin < level
            pushes = (margin == level) if integral else np.zeros_like(wins)
        total += weight * _leg_ev(wins, pushes, hk_price, shift)
    return total


def total_ev(runs: np.ndarray, line: AsianLine, *, hk_price: float,
             over: bool, shift: float = 0.0) -> float:
    """EV of one unit on an Asian total."""
    total = 0.0
    for level, weight in ((line.low, 1.0 - line.weight),
                          (line.high, line.weight)):
        if weight == 0.0:
            continue
        integral = float(level).is_integer()
        wins = runs > level if over else runs < level
        pushes = (runs == level) if integral else np.zeros_like(wins)
        total += weight * _leg_ev(wins, pushes, hk_price, shift)
    return total


def _solve_fair(gap_at_line, lo: float, hi: float, step: float = 0.5) -> float:
    """Line at which both sides of a market are an even proposition.

    Scans half-run lines for a sign change, then refines inside the bracket
    using the split weight, which is continuous even though scores are not.

    The scan is the point. Bisecting on the line directly cannot work --
    `runs > line` only changes at integers, so the search lands on one and
    stops. Nor can the bracket be assumed: a window guessed from the mean can
    sit entirely below the answer, and then the refinement saturates at its
    own edge and returns a boundary that looks like a real number. Both of
    those produced clean-looking integers here before this was rewritten.
    """
    grid = np.arange(lo, hi + step, step)
    values = [gap_at_line(float(x)) for x in grid]

    crossing = None
    for i in range(len(grid) - 1):
        if values[i] >= 0 >= values[i + 1]:
            crossing = i
            break
    if crossing is None:
        # No sign change anywhere in the window: report the closest point
        # rather than a boundary that would masquerade as a solution.
        return float(grid[int(np.argmin(np.abs(values)))])

    low, high = float(grid[crossing]), float(grid[crossing + 1])
    a, b = 0.0, 1.0
    for _ in range(30):
        mid = 0.5 * (a + b)
        if gap_at_line(low + mid * (high - low)) > 0:
            a = mid
        else:
            b = mid
    return low + 0.5 * (a + b) * (high - low)


def _as_line(value: float) -> AsianLine:
    """Represent an arbitrary line as a split between neighbouring halves."""
    low = np.floor(value * 2.0) / 2.0
    weight = (value - low) / 0.5
    return AsianLine(float(low), float(low) + 0.5, float(weight), "")


def fair_total(runs: np.ndarray, hk_price: float = 1.0) -> float:
    """The total at which over and under are an even proposition."""
    def gap(line: float) -> float:
        obj = _as_line(line)
        return (total_ev(runs, obj, hk_price=hk_price, over=True)
                - total_ev(runs, obj, hk_price=hk_price, over=False))
    return _solve_fair(gap, 0.0, 25.0)


def fair_handicap(margin: np.ndarray, hk_price: float = 1.0) -> float:
    """The handicap at which laying and receiving are an even proposition."""
    def gap(line: float) -> float:
        obj = _as_line(line)
        return (handicap_ev(margin, obj, hk_price=hk_price, laying=True)
                - handicap_ev(margin, obj, hk_price=hk_price, laying=False))
    return _solve_fair(gap, -3.0, 8.0)
