#!/usr/bin/env python3
"""Scan the MLB run line and total markets and emit a markdown report.

  python3 scripts/scan_market.py --fetch --out reports/today.md
  python3 scripts/scan_market.py --snapshot data/snapshots/<file>.json

The report deliberately separates two things that look alike on a screen:
the market's own de-vigged consensus, which we can measure, and a model
probability, which requires handicapping inputs. When those inputs are
absent the report says so and recommends nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlbline.devig import decimal_to_american  # noqa: E402
from mlbline.market import book_holds, scan  # noqa: E402
from mlbline.odds import fetch_odds, load_snapshot, normalise  # noqa: E402
from mlbline.policy import (  # noqa: E402
    BettingPolicy,
    DataCompleteness,
    evaluate,
)

EASTERN = ZoneInfo("America/New_York")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--fetch", action="store_true",
                     help="pull live odds using ODDS_API_KEY")
    src.add_argument("--snapshot", type=str,
                     help="path to a saved raw odds payload")
    p.add_argument("--out", type=str, default=None,
                   help="write markdown here instead of stdout")
    p.add_argument("--save-snapshot", type=str, default=None,
                   help="with --fetch, also save the raw payload here")
    p.add_argument("--devig", default="shin",
                   choices=["shin", "multiplicative", "additive", "power"])
    p.add_argument("--all-lines", action="store_true",
                   help="include mirror run lines and alternate totals")
    return p.parse_args()


def et(dt: datetime) -> str:
    return dt.astimezone(EASTERN).strftime("%-I:%M %p ET")


def render(games, rows, holds, policy, completeness, devig_method) -> str:
    now = datetime.now(timezone.utc)
    sides = [(r, s) for r in rows for s in r.sides]
    sides.sort(key=lambda t: -t[1].ev_vs_consensus)

    out: list[str] = []
    w = out.append

    w("# MLB Run Line & Total — Market Scan")
    w("")
    w(f"- Snapshot taken: **{now:%Y-%m-%d %H:%M} UTC** ({et(now)})")
    w(f"- Games priced: **{len(games)}**   Lines analysed: **{len(rows)}**")
    w(f"- De-vig method: **{devig_method}**")
    w("")

    missing = completeness.missing()
    w("## Data completeness")
    w("")
    if missing:
        w("Handicapping inputs **not** obtained for this slate:")
        w("")
        for field_name in missing:
            w(f"- `{field_name}`")
        w("")
        w("Without these no run distribution can be estimated, so no model "
          "probability exists and nothing below is a bet recommendation.")
    else:
        w("All required inputs present.")
    w("")

    w("## Bookmaker margin (median hold, totals)")
    w("")
    w("| Book | Hold |")
    w("|---|---|")
    for book, hold in sorted(holds.items(), key=lambda kv: kv[1]):
        w(f"| {book} | {hold * 100:.2f}% |")
    w("")
    w("The lowest-hold books supply the best price on most lines simply "
      "because they charge less vig. That is a cheaper ticket, not an edge.")
    w("")

    w("## Best available price vs market consensus")
    w("")
    w("`Consensus` is the median de-vigged probability across every *other* "
      "book at the same line, so a lone outlier cannot set its own baseline. "
      "`Edge` is how far the best takeable price beats that consensus.")
    w("")
    w("| Game | First pitch | Side | Best price | Book | Consensus | "
      "Fair | Edge | Books |")
    w("|---|---|---|---|---|---|---|---|---|")
    for r, s in sides:
        fair = decimal_to_american(1.0 / s.loo_consensus_prob)
        flag = " ⚠︎" if s.best_is_stale else ""
        w(f"| {r.matchup} | {et(r.commence_time)} | {s.side} | "
          f"{s.best_american:+.0f}{flag} | {s.best_book} | "
          f"{s.loo_consensus_prob * 100:.2f}% | {fair:+.0f} | "
          f"{s.ev_vs_consensus * 100:+.2f}% | {s.n_books} |")
    w("")

    w("## Verdict")
    w("")
    qualifying = []
    for r, s in sides:
        gate = evaluate(
            policy=policy,
            completeness=completeness,
            # No handicapping inputs means no model probability. Passing the
            # market's own number here would be circular, so it stays None.
            model_probability=None,
            market_probability=s.loo_consensus_prob,
            decimal_odds=s.best_decimal,
            n_books=s.n_books,
        )
        if gate.passed:
            qualifying.append((r, s, gate))

    best = sides[0][1] if sides else None
    if best is not None:
        w(f"Best market-relative edge on the board: **{best.side}** at "
          f"{best.best_american:+.0f} ({best.best_book}), "
          f"**{best.ev_vs_consensus * 100:+.2f}%** against consensus — "
          f"versus a **{policy.min_expected_value * 100:.0f}%** minimum.")
        w("")
    cleared = sum(1 for _, s in sides
                  if s.ev_vs_consensus >= policy.min_expected_value)
    w(f"Sides clearing the +{policy.min_expected_value * 100:.0f}% threshold: "
      f"**{cleared} of {len(sides)}**.")
    w("")

    if qualifying:
        w("### Recommended")
        w("")
        for r, s, gate in qualifying:
            w(f"- **{r.matchup}** — {s.side} {s.best_american:+.0f} "
              f"({s.best_book}), stake {gate.stake_fraction * 100:.2f}% "
              "of bankroll")
    else:
        w("### 不下注 — NO BET")
        w("")
        w("No selection on this slate satisfies the policy. The blocking "
          "reasons are structural, not marginal:")
        w("")
        sample = evaluate(
            policy=policy, completeness=completeness,
            model_probability=None,
            market_probability=0.5, decimal_odds=2.0, n_books=16,
        )
        for reason in sample.reasons:
            w(f"- {reason}")
    w("")
    return "\n".join(out)


def main() -> int:
    args = parse_args()

    if args.fetch:
        payload = fetch_odds()
        if args.save_snapshot:
            Path(args.save_snapshot).parent.mkdir(parents=True, exist_ok=True)
            Path(args.save_snapshot).write_text(
                json.dumps(payload, indent=2), encoding="utf-8")
    else:
        payload = load_snapshot(args.snapshot)

    games = normalise(payload)
    rows = scan(games, markets=("spreads", "totals"),
                devig_method=args.devig, all_lines=args.all_lines)
    holds = book_holds(games, "totals")

    # Market prices are the only input this environment can reach; every
    # baseball data source is blocked by egress policy.
    completeness = DataCompleteness(market_prices=bool(games))

    report = render(games, rows, holds, BettingPolicy(), completeness,
                    args.devig)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
