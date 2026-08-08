"""Consensus pricing, line shopping, and market-relative expected value.

A note on what the EV in this module does and does not mean.

`ev_vs_consensus` compares the best price a bettor can actually reach against
the de-vigged consensus of every *other* book at the same line. It answers
"is this book out of step with the market?" It does not answer "is the market
wrong?" -- that requires an independent estimate of the run distribution,
which lives in `rundist` and needs pitcher, lineup, park and weather inputs.

The two are not interchangeable and must never be reported as if they were.
Market-relative EV is real but small and fragile; it decays as books correct,
and it is the reason line shopping pays, not the reason handicapping pays.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import datetime, timezone

from .devig import (
    decimal_to_american,
    devig,
    method_spread,
    overround,
    probability_to_american,
)
from .odds import Game, Quote, pair_sides

# Quotes older than this are reported but excluded from "best price", since a
# stale screen price is not something a bettor can reliably take.
STALE_AFTER_SECONDS = 45 * 60


@dataclass
class SideSummary:
    """Consensus and best available price for one side of one line."""

    side: str
    n_books: int
    consensus_prob: float          # median de-vigged probability, all books
    prob_min: float
    prob_max: float
    prob_stdev: float
    method_risk: float             # spread across de-vig conventions
    best_book: str
    best_american: float
    best_decimal: float
    best_is_stale: bool
    # Benchmark excluding the book offering the best price, so an outlier is
    # not allowed to drag its own baseline toward itself.
    loo_consensus_prob: float
    ev_vs_consensus: float         # loo_consensus_prob * best_decimal - 1

    @property
    def fair_american(self) -> float:
        return probability_to_american(self.loo_consensus_prob)

    @property
    def min_acceptable_american(self) -> float:
        """Break-even price against the leave-one-out consensus."""
        return probability_to_american(self.loo_consensus_prob)

    @property
    def edge_over_method_risk(self) -> float:
        """EV expressed in units of de-vig convention uncertainty.

        Below roughly 1.0 the apparent edge is inside the noise created by
        choosing one de-vig method over another.
        """
        if self.method_risk <= 0:
            return float("inf")
        return self.ev_vs_consensus / self.method_risk


@dataclass
class LineSummary:
    game_id: str
    matchup: str
    commence_time: datetime
    market: str
    point: float | None
    sides: list[SideSummary]
    median_hold: float             # typical book margin on this line

    def side(self, name: str) -> SideSummary | None:
        for s in self.sides:
            if s.side == name:
                return s
        return None

    @property
    def best_ev(self) -> float:
        return max((s.ev_vs_consensus for s in self.sides), default=0.0)


def book_holds(games: list[Game], market: str) -> dict[str, float]:
    """Median overround per book across the slate.

    Used to identify low-margin books empirically rather than by reputation.
    The lowest-hold books are the most useful consensus anchors.
    """
    collected: dict[str, list[float]] = {}
    for game in games:
        point = game.modal_point(market) if market != "h2h" else None
        quotes = game.select(market, point)
        for book, a, b in pair_sides(quotes):
            collected.setdefault(book, []).append(
                overround([a.decimal, b.decimal])
            )
    return {b: statistics.median(v) for b, v in sorted(collected.items())}


def summarise_line(game: Game, market: str, point: float | None,
                   *, devig_method: str = "shin",
                   now: datetime | None = None) -> LineSummary | None:
    """Build the consensus / best-price picture for one line of one game."""
    now = now or datetime.now(timezone.utc)
    quotes = game.select(market, point)
    if not quotes:
        return None

    # De-vig every book that prices both sides.
    fair_by_side: dict[str, list[float]] = {}
    holds: list[float] = []
    method_risk_by_side: dict[str, list[float]] = {}
    best_by_side: dict[str, Quote] = {}

    for _book, a, b in pair_sides(quotes):
        decimals = [a.decimal, b.decimal]
        probs = devig(decimals, devig_method)
        holds.append(overround(decimals))
        for idx, q in enumerate((a, b)):
            fair_by_side.setdefault(q.label, []).append(probs[idx])
            method_risk_by_side.setdefault(q.label, []).append(
                method_spread(decimals, idx)
            )

    if len(fair_by_side) != 2:
        return None

    # Best takeable price per side, ignoring stale quotes.
    for q in quotes:
        fresh = q.age_seconds(now) <= STALE_AFTER_SECONDS
        incumbent = best_by_side.get(q.label)
        if incumbent is None:
            best_by_side[q.label] = q
            continue
        incumbent_fresh = incumbent.age_seconds(now) <= STALE_AFTER_SECONDS
        # Prefer fresh over stale, then higher decimal odds.
        if (fresh, q.decimal) > (incumbent_fresh, incumbent.decimal):
            best_by_side[q.label] = q

    sides: list[SideSummary] = []
    for side, probs in fair_by_side.items():
        best = best_by_side[side]

        # Leave-one-out consensus: re-de-vig excluding the best-price book.
        loo_probs = []
        for book, a, b in pair_sides(quotes):
            if book == best.book:
                continue
            pair_probs = devig([a.decimal, b.decimal], devig_method)
            for idx, q in enumerate((a, b)):
                if q.label == side:
                    loo_probs.append(pair_probs[idx])
        loo = statistics.median(loo_probs) if loo_probs else statistics.median(probs)

        sides.append(SideSummary(
            side=side,
            n_books=len(probs),
            consensus_prob=statistics.median(probs),
            prob_min=min(probs),
            prob_max=max(probs),
            prob_stdev=statistics.pstdev(probs) if len(probs) > 1 else 0.0,
            method_risk=statistics.median(method_risk_by_side[side]),
            best_book=best.book,
            best_american=best.american,
            best_decimal=best.decimal,
            best_is_stale=best.age_seconds(now) > STALE_AFTER_SECONDS,
            loo_consensus_prob=loo,
            ev_vs_consensus=loo * best.decimal - 1.0,
        ))

    sides.sort(key=lambda s: -s.ev_vs_consensus)
    return LineSummary(
        game_id=game.game_id,
        matchup=game.matchup,
        commence_time=game.commence_time,
        market=market,
        point=point,
        sides=sides,
        median_hold=statistics.median(holds) if holds else float("nan"),
    )


def scan(games: list[Game], *, markets: tuple[str, ...] = ("spreads", "totals"),
         devig_method: str = "shin",
         min_books: int = 3,
         all_lines: bool = False,
         now: datetime | None = None) -> list[LineSummary]:
    """Summarise each requested market for every game.

    By default only the most widely offered line per market is returned,
    which is the one the market has actually settled on. `all_lines` also
    returns the mirror run line and any alternate totals, priced by fewer
    books and correspondingly noisier.
    """
    out: list[LineSummary] = []
    for game in games:
        for market in markets:
            if market == "h2h":
                points: list[float | None] = [None]
            else:
                offered = game.offered_points(market)
                points = list(offered) if all_lines else offered[:1]
            for point in points:
                summary = summarise_line(game, market, point,
                                         devig_method=devig_method, now=now)
                if summary is None:
                    continue
                if min(s.n_books for s in summary.sides) < min_books:
                    continue
                out.append(summary)
    return out


def kelly_fraction(prob: float, decimal_odds: float,
                   multiplier: float = 0.25) -> float:
    """Fractional Kelly stake as a share of bankroll.

    Returns 0.0 when the bet is not advantaged. `multiplier` defaults to
    quarter Kelly.
    """
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    edge = prob * b - (1.0 - prob)
    if edge <= 0:
        return 0.0
    return multiplier * edge / b


def format_american(value: float) -> str:
    return f"{value:+.0f}"


def fair_price_string(prob: float) -> str:
    return format_american(decimal_to_american(1.0 / prob))
