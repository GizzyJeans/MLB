#!/usr/bin/env python3
"""Price each run line off its own game's moneyline and total.

  python3 scripts/relative_value.py --snapshot data/snapshots/<file>.json

Reports where the quoted run line disagrees with the price implied by the
other two markets. This is relative value: it assumes the moneyline and total
are correct and asks whether the run line has kept up. It is not a substitute
for handicapping, and the report labels it accordingly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlbline.devig import decimal_to_american  # noqa: E402
from mlbline.implied import solve  # noqa: E402
from mlbline.market import summarise_line  # noqa: E402
from mlbline.odds import load_snapshot, normalise  # noqa: E402
from mlbline.policy import BettingPolicy, DataCompleteness, evaluate  # noqa: E402

# Above this residual against the anchor markets the solve is not trustworthy
# and the game is reported as unsolved rather than quietly priced.
MAX_FIT_ERROR = 0.01


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--bankroll", type=float, default=10_000.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    games = normalise(load_snapshot(args.snapshot))
    policy = BettingPolicy()
    # Market prices are reachable; no baseball data source is.
    completeness = DataCompleteness(market_prices=True)

    header = (f"{'GAME':38s} {'T':5s} {'預測比分':11s} {'側':26s} "
              f"{'模型':7s} {'市場':7s} {'差':8s} {'最佳價':11s} {'EV':7s}")
    print(header)
    print("-" * 132)

    rows = []
    for game in games:
        line = game.modal_point("totals")
        # The run line must be taken at whichever side the market actually
        # hangs it from. Assuming the home team lays the runs silently drops
        # every game with a road favourite, which is a biased sample rather
        # than a smaller one.
        run_line = game.modal_point("spreads")
        h2h = summarise_line(game, "h2h", None)
        totals = summarise_line(game, "totals", line)
        spreads = (summarise_line(game, "spreads", run_line)
                   if run_line is not None else None)
        if not (h2h and totals and spreads):
            print(f"{game.matchup[:37]:38s} — 市場不完整，跳過")
            continue

        home_win = next(s for s in h2h.sides
                        if s.side == game.home_team).loo_consensus_prob
        over = next(s for s in totals.sides
                    if s.side.startswith("Over")).loo_consensus_prob

        implied = solve(game.matchup, market_home_win=home_win,
                        market_over=over, total_line=line)

        if implied.fit_error > MAX_FIT_ERROR:
            print(f"{game.matchup[:37]:38s} — 解不收斂 "
                  f"(殘差 {implied.fit_error * 100:.2f}pp)，跳過")
            continue

        sim = implied.simulation
        for label, model_p in (
            (f"{game.home_team} {run_line:+g}",
             sim.prob_cover(run_line, home=True)),
            (f"{game.away_team} {-run_line:+g}",
             sim.prob_cover(-run_line, home=False)),
        ):
            side = next((s for s in spreads.sides if s.side == label), None)
            if side is None:
                continue
            divergence = model_p - side.loo_consensus_prob
            ev = model_p * side.best_decimal - 1.0
            rows.append((game, implied, side, model_p, divergence, ev))
            print(f"{game.matchup[:37]:38s} {line:5.1f} "
                  f"{implied.predicted_score:11s} {label[:25]:26s} "
                  f"{model_p * 100:6.2f}% {side.loo_consensus_prob * 100:6.2f}% "
                  f"{divergence * 100:+7.2f}pp "
                  f"{side.best_american:+5.0f} {side.best_book[:5]:5s} "
                  f"{ev * 100:+6.2f}%")

    print()
    rows.sort(key=lambda r: -r[5])
    passed = []
    for game, implied, side, model_p, divergence, ev in rows:
        gate = evaluate(
            policy=policy, completeness=completeness,
            model_probability=model_p,
            market_probability=side.loo_consensus_prob,
            decimal_odds=side.best_decimal, n_books=side.n_books,
        )
        if gate.passed:
            passed.append((game, side, model_p, ev, gate))

    best = rows[0] if rows else None
    if best:
        print(f"最大相對價值: {best[2].side} @ {best[2].best_american:+.0f} "
              f"({best[2].best_book})  模型 {best[3] * 100:.2f}% vs 市場 "
              f"{best[2].loo_consensus_prob * 100:.2f}%  "
              f"差 {best[4] * 100:+.2f}pp  EV {best[5] * 100:+.2f}%")
    print(f"通過政策門檻: {len(passed)} / {len(rows)}")
    if not passed:
        print("\n不下注 — NO BET")
        sample = evaluate(
            policy=policy, completeness=completeness,
            model_probability=0.60, market_probability=0.50,
            decimal_odds=2.10, n_books=16,
        )
        for reason in sample.reasons:
            print(f"  - {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
