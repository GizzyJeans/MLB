"""Market self-verification for handicap candidates (gate 2B).

Gate 2 asks the model to disagree with the US market by three points. The
model is fitted to that market, so the gate almost never opens -- and by
construction it stays shut on every edge that comes from the Asian board
hanging a different number than the US books do. Those edges need no opinion
from the model at all. They need the US market to confirm them.

Two US lines are enough to price any handicap whose legs sit at 0, 0.5, 1 or
1.5 runs, with no simulation anywhere:

    moneyline        P(favourite wins)          = P(margin >= 1)
    run line -1.5    P(favourite wins by 2+)    = P(margin >= 2)

Their difference is the chance of a one-run favourite win, which is exactly
the outcome an integer leg pushes on. A leg at 2 or more would need the
two-run margin, which the US main market does not price, so those lines are
not eligible. Totals are not eligible either: a board total that differs
from the US number needs the probability of landing exactly on an integer,
which only alternate totals price, and US books rarely hang enough of them.

The rule and every threshold are declared in docs/gate_2b.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median

from .asian import AsianLine
from .devig import METHODS
from .odds import Game

# Declared in docs/gate_2b.md before any retrospective scan. Not tunable.
MIN_BOOKS = 8            # quoting both sides, for each of the two US lines
MARKET_EV_MIN = 0.04     # same bar as gate 1, under every de-vig method
BOOK_AGREEMENT = 0.75    # share of books whose own prices give EV > 0
ELIGIBLE_LEVELS = (0.0, 0.5, 1.0, 1.5)
BOOK_METHOD = "shin"     # the project's default de-vig, for per-book checks

# Representative favourite margins for the three outcomes the two US lines
# can tell apart. Exact for any leg at 1.5 or less.
REGIONS = {"loses": -1, "by_one": 1, "by_two_plus": 2}


def _legs(line: AsianLine):
    for level, weight in ((line.low, 1.0 - line.weight),
                          (line.high, line.weight)):
        if weight > 0.0:
            yield level, weight


def _leg_outcomes(level: float, laying: bool, win: float,
                  two_plus: float) -> tuple[float, float, float]:
    """(p_win, p_push, p_lose) for one leg."""
    one = win - two_plus
    if level < 1.0:                       # 0 or 0.5: the moneyline
        fav = (win, 0.0, 1.0 - win)
    elif level == 1.0:                    # pushes on a one-run win
        fav = (two_plus, one, 1.0 - win)
    else:                                 # 1.5
        fav = (two_plus, 0.0, 1.0 - two_plus)
    return fav if laying else (fav[2], fav[1], fav[0])


def ticket_ev(line: AsianLine, laying: bool, price: float, win: float,
              two_plus: float) -> float:
    """EV of one unit on the board ticket, from market probabilities."""
    total = 0.0
    for level, weight in _legs(line):
        p_win, _, p_lose = _leg_outcomes(level, laying, win, two_plus)
        total += weight * (p_win * price - p_lose)
    return total


def required_price(line: AsianLine, laying: bool, win: float,
                   two_plus: float, target: float) -> float:
    """Hong Kong price at which the ticket's EV equals `target`."""
    gain = loss = 0.0
    for level, weight in _legs(line):
        p_win, _, p_lose = _leg_outcomes(level, laying, win, two_plus)
        gain += weight * p_win
        loss += weight * p_lose
    return (target + loss) / gain


def ticket_returns(line: AsianLine, laying: bool,
                   price: float) -> dict[str, float]:
    """Return per unit staked in each of the three margin outcomes."""
    out = {}
    for name, margin in REGIONS.items():
        ret = 0.0
        for level, weight in _legs(line):
            won = margin > level if laying else margin < level
            push = margin == level
            ret += weight * (price if won else (0.0 if push else -1.0))
        out[name] = ret
    return out


def kelly(line: AsianLine, laying: bool, price: float, win: float,
          two_plus: float) -> float:
    """Growth-optimal stake fraction for the split ticket."""
    probs = {"loses": 1.0 - win, "by_one": win - two_plus,
             "by_two_plus": two_plus}
    rets = ticket_returns(line, laying, price)

    def slope(f: float) -> float:
        return sum(probs[k] * rets[k] / (1.0 + f * rets[k]) for k in probs)

    if slope(0.0) <= 0.0:
        return 0.0
    lo, hi = 0.0, 0.999
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if slope(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return lo


@dataclass
class MarketCheck:
    """Everything the US market says about one board handicap ticket."""

    eligible: bool
    reason: str = ""
    n_ml: int = 0
    n_rl: int = 0
    win: dict[str, float] = field(default_factory=dict)       # by method
    two_plus: dict[str, float] = field(default_factory=dict)  # by method
    ev: dict[str, float] = field(default_factory=dict)        # by method
    book_evs: list[tuple[str, float]] = field(default_factory=list)
    min_price: float | None = None

    @property
    def ev_min(self) -> float:
        return min(self.ev.values()) if self.ev else float("nan")

    @property
    def worst_method(self) -> str:
        return min(self.ev, key=self.ev.get) if self.ev else "-"

    @property
    def agreement(self) -> float:
        if not self.book_evs:
            return 0.0
        return sum(1 for _, ev in self.book_evs if ev > 0.0) / len(self.book_evs)

    @property
    def failures(self) -> list[str]:
        """Which declared conditions fail; empty means the market confirms."""
        if not self.eligible:
            return [self.reason]
        out = []
        if self.n_ml < MIN_BOOKS or self.n_rl < MIN_BOOKS:
            out.append(f"書商不足（獨贏 {self.n_ml}、讓分 {self.n_rl}）")
        if len(self.book_evs) < MIN_BOOKS:
            out.append(f"兩條都報的書商只有 {len(self.book_evs)} 家")
        if not self.ev_min >= MARKET_EV_MIN:
            out.append(f"市場 EV 最差 {self.ev_min * 100:+.2f}%"
                       f"（{self.worst_method}）")
        if self.agreement < BOOK_AGREEMENT:
            out.append(f"書商同意 {self.agreement * 100:.0f}%")
        return out

    @property
    def passes(self) -> bool:
        return not self.failures


def check_handicap(game: Game, favourite: str, line: AsianLine, laying: bool,
                   price: float, reading_conflict: bool = False) -> MarketCheck:
    """Price one board handicap ticket from US moneyline and run line only."""
    if reading_conflict:
        return MarketCheck(False, "讀法衝突，方向不確定")
    if any(level not in ELIGIBLE_LEVELS for level, _ in _legs(line)):
        return MarketCheck(False, "有一腳超過 1.5，美盤主盤無法定價")

    underdog = game.away_team if favourite == game.home_team else game.home_team
    # Run lines are keyed from the home side, so the favourite's -1.5 is
    # -1.5 when it is at home and +1.5 when it is on the road.
    rl_key = -1.5 if favourite == game.home_team else 1.5

    def by_book(quotes) -> dict[str, dict[str, float]]:
        books: dict[str, dict[str, float]] = {}
        for q in quotes:
            books.setdefault(q.book, {})[q.side] = q.decimal
        out = {}
        for book, sides in books.items():
            if favourite in sides and underdog in sides:
                pair = [sides[favourite], sides[underdog]]
                out[book] = {m: fn(pair)[0] for m, fn in METHODS.items()}
        return out

    ml = by_book(game.select("h2h"))
    rl = by_book(game.select("spreads", point=rl_key))
    check = MarketCheck(True, n_ml=len(ml), n_rl=len(rl))
    if not ml or not rl:
        check.eligible = False
        check.reason = "美盤缺獨贏或 −1.5 讓分盤"
        return check

    for method in METHODS:
        win = median(b[method] for b in ml.values())
        two = median(b[method] for b in rl.values())
        if two > win:
            check.eligible = False
            check.reason = "美盤讓分盤與獨贏互相矛盾"
            return check
        check.win[method], check.two_plus[method] = win, two
        check.ev[method] = ticket_ev(line, laying, price, win, two)

    for book in sorted(set(ml) & set(rl)):
        win, two = ml[book][BOOK_METHOD], rl[book][BOOK_METHOD]
        if two <= win:
            check.book_evs.append(
                (book, ticket_ev(line, laying, price, win, two)))

    check.min_price = max(
        required_price(line, laying, check.win[m], check.two_plus[m],
                       MARKET_EV_MIN)
        for m in METHODS)
    return check
