"""Odds API client and normalisation into flat quotes.

The upstream payload nests game -> bookmaker -> market -> outcome, which is
awkward for the comparisons we actually want (all books at one line, one
market, one game). Everything here flattens to `Quote` rows and keeps the
per-book `last_update` attached, since a stale quote is not a real price.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Iterable, Iterator, Sequence

from .devig import american_to_decimal

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "baseball_mlb"


@dataclass(frozen=True)
class Quote:
    """A single price, from one book, for one side of one line."""

    game_id: str
    book: str
    market: str          # "h2h" | "spreads" | "totals"
    side: str            # team name, or "Over" / "Under"
    point: float | None  # run line or total; None for moneyline
    american: float
    last_update: datetime
    # Identifier for the two-sided market this quote belongs to. Totals show
    # the same number on both sides, so it is just the total. Run lines do
    # not: a book quoting "Mariners -1.5 / Rays +1.5" and one quoting
    # "Mariners +1.5 / Rays -1.5" are pricing two *different* markets, and
    # pooling them silently mixes a ~35% outcome with a ~65% one. Referencing
    # the point to the home team keeps them apart while still pairing the two
    # halves of each.
    line_key: float | None = None

    @property
    def decimal(self) -> float:
        return american_to_decimal(self.american)

    @property
    def label(self) -> str:
        """Human-readable side, including the line it is quoted at."""
        if self.market == "totals":
            return f"{self.side} {self.point:g}"
        if self.market == "spreads":
            return f"{self.side} {self.point:+g}"
        return self.side

    def age_seconds(self, now: datetime | None = None) -> float:
        now = now or datetime.now(timezone.utc)
        return (now - self.last_update).total_seconds()


@dataclass
class Game:
    game_id: str
    commence_time: datetime
    home_team: str
    away_team: str
    quotes: list[Quote] = field(default_factory=list)

    @property
    def matchup(self) -> str:
        return f"{self.away_team} @ {self.home_team}"

    def select(self, market: str, point: float | None = None,
               side: str | None = None) -> list[Quote]:
        """Quotes filtered by market, and optionally exact line and side.

        Lines are matched on `line_key`, so asking for the 1.5 run line
        returns both the -1.5 favourite and the +1.5 underdog.
        """
        out = [q for q in self.quotes if q.market == market]
        if point is not None:
            out = [q for q in out if q.line_key is not None
                   and abs(q.line_key - point) < 1e-9]
        if side is not None:
            out = [q for q in out if q.side == side]
        return out

    def offered_points(self, market: str) -> list[float]:
        """Distinct lines offered for a market, most common first."""
        counts: dict[float, int] = {}
        for q in self.quotes:
            if q.market == market and q.line_key is not None:
                counts[q.line_key] = counts.get(q.line_key, 0) + 1
        return [p for p, _ in sorted(counts.items(),
                                     key=lambda kv: (-kv[1], kv[0]))]

    def modal_point(self, market: str) -> float | None:
        """The line the market has settled on, i.e. the most widely offered."""
        points = self.offered_points(market)
        return points[0] if points else None


def _parse_ts(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def eastern_date(moment: datetime) -> str:
    """Calendar date of a game in US Eastern time.

    A slate spans two UTC dates -- a 10pm Eastern first pitch is already
    tomorrow in UTC -- so the UTC date splits one night's games in two.
    Eastern is what a schedule means by "the 14th".
    """
    return moment.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")


def select_slate(games: list[Game], date: str | None = None) -> dict[str, Game]:
    """Index games by matchup for one slate.

    Teams play series, so the same pairing appears on consecutive days. An
    odds feed pulled late in the evening carries tomorrow's lines alongside
    tonight's, and indexing by matchup alone lets the later game overwrite
    the earlier one -- silently pricing tomorrow's pitchers against tonight's
    board. Filtering to the slate's own date prevents that.

    A pairing can also legitimately repeat within one date, as a
    doubleheader. Those are keyed "<matchup> G1" and "<matchup> G2" in start
    order, so both remain reachable and neither can stand in for the other.
    A caller asking for the bare matchup of a doubleheader gets nothing back
    rather than an arbitrary half of it.
    """
    grouped: dict[str, list[tuple[str, Game]]] = {}
    for game in games:
        day = eastern_date(game.commence_time)
        if date and day != date:
            continue
        grouped.setdefault(game.matchup, []).append((day, game))

    chosen: dict[str, Game] = {}
    for matchup, entries in grouped.items():
        days = {day for day, _ in entries}
        if len(days) > 1:
            # Two dates is a series, not a doubleheader. Numbering them G1
            # and G2 would quietly pair tomorrow's lines with tonight's
            # board, which is the failure this guard exists to catch.
            raise ValueError(
                f"{matchup} appears on {sorted(days)}. Pass a date to choose "
                "a slate."
            )
        group = [game for _, game in entries]
        if len(group) == 1:
            chosen[matchup] = group[0]
            continue
        for number, game in enumerate(
                sorted(group, key=lambda g: g.commence_time), start=1):
            chosen[f"{matchup} G{number}"] = game
    return chosen


def _line_key(market: str, point: float | None, side: str,
              home_team: str) -> float | None:
    """Market identifier shared by both sides of one line.

    For spreads the point is expressed from the home team's side, so
    "home -1.5" and "away +1.5" collapse to -1.5 while the mirror market
    ("home +1.5" / "away -1.5") stays separate at +1.5.
    """
    if point is None:
        return None
    if market != "spreads":
        return point
    return point if side == home_team else -point


def fetch_odds(api_key: str | None = None, *,
               regions: str = "us,us2",
               markets: str = "h2h,spreads,totals",
               timeout: int = 60) -> list[dict]:
    """Pull the live board. Raises if no API key is configured."""
    api_key = api_key or os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No Odds API key. Set ODDS_API_KEY in the environment."
        )
    query = urllib.parse.urlencode({
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american",
        "dateFormat": "iso",
    })
    url = f"{ODDS_API_BASE}/sports/{SPORT_KEY}/odds/?{query}"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_snapshot(path: str) -> list[dict]:
    """Load a previously saved raw payload."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def normalise(payload: Sequence[dict]) -> list[Game]:
    """Flatten the nested API payload into `Game` objects holding `Quote`s."""
    games: list[Game] = []
    for raw in payload:
        game = Game(
            game_id=raw["id"],
            commence_time=_parse_ts(raw["commence_time"]),
            home_team=raw["home_team"],
            away_team=raw["away_team"],
        )
        for book in raw.get("bookmakers", []):
            for market in book.get("markets", []):
                updated = _parse_ts(
                    market.get("last_update") or book["last_update"]
                )
                for outcome in market.get("outcomes", []):
                    point = outcome.get("point")
                    game.quotes.append(Quote(
                        game_id=game.game_id,
                        book=book["key"],
                        market=market["key"],
                        side=outcome["name"],
                        point=point,
                        american=float(outcome["price"]),
                        last_update=updated,
                        line_key=_line_key(
                            market["key"], point,
                            outcome["name"], game.home_team),
                    ))
        games.append(game)
    return games


def pair_sides(quotes: Iterable[Quote]) -> Iterator[tuple[str, Quote, Quote]]:
    """Yield (book, side_a, side_b) for books quoting both sides of a line.

    Books showing only one side cannot be de-vigged and are dropped, which is
    the correct behaviour: a one-sided quote carries no margin information.
    """
    by_book: dict[str, list[Quote]] = {}
    for q in quotes:
        by_book.setdefault(q.book, []).append(q)
    for book, pair in sorted(by_book.items()):
        if len(pair) == 2:
            yield book, pair[0], pair[1]
