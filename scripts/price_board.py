#!/usr/bin/env python3
"""Price an Asian board's handicap and total lines at the prices it quotes.

  python3 scripts/price_board.py --board data/boards/<file>.json \
      --snapshot data/snapshots/<file>.json

Expected runs per side come from the game's own moneyline and total in the US
market, so the question is narrow: given what the wider market thinks of this
game, are this board's lines priced correctly at the flat prices it charges?

Two things are corrected before anything is ranked.

The engine understates each side's share of wins by two or more, measured
against 7,289 real games and against this slate's own market. Uncorrected it
makes every underdog receiving runs look valuable, which is a property of the
model rather than of any game.

The board's margin comes off the top: 0.950 on a handicap is a 2.6% hold,
0.940 on a total is 3.1%. A correctly decoded, fairly hung line should
therefore price at roughly -2.5% and -3.0% for *both* sides. That expectation
is the sanity check -- anything far outside it means the line was decoded
wrong, not that an edge was found.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlbline.asian import (  # noqa: E402
    fair_handicap,
    fair_total,
    handicap_ev,
    parse_line,
    total_ev,
)
from mlbline.implied import solve  # noqa: E402
from mlbline.market import summarise_line  # noqa: E402
from mlbline.odds import load_snapshot, normalise  # noqa: E402

# Measured shortfall in the favourite's cover probability, from the slate's
# mean disagreement with the market and consistent with the out-of-sample
# check against real games.
COVER_BIAS = 0.0128


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    board = json.loads(Path(args.board).read_text(encoding="utf-8"))
    hk_h, hk_t = board["handicap_price_hk"], board["total_price_hk"]
    games = {g.matchup: g for g in normalise(load_snapshot(args.snapshot))}

    candidates, checks = [], []

    for entry in board["games"]:
        matchup = f"{entry['away']} @ {entry['home']}"
        game = games.get(matchup)
        if game is None:
            continue
        us_total = game.modal_point("totals")
        h2h = summarise_line(game, "h2h", None)
        totals = summarise_line(game, "totals", us_total)
        if not (h2h and totals):
            print(f"{matchup}: 市場不完整，跳過")
            continue

        home_win = next(s for s in h2h.sides
                        if s.side == game.home_team).loo_consensus_prob
        over = next(s for s in totals.sides
                    if s.side.startswith("Over")).loo_consensus_prob
        sim = solve(matchup, market_home_win=home_win, market_over=over,
                    total_line=us_total).simulation

        favourite = entry["favourite"] or game.home_team
        fav_home = favourite == game.home_team
        margin = sim.margin if fav_home else -sim.margin
        underdog = game.away_team if fav_home else game.home_team

        hcap, tline = parse_line(entry["handicap"]), parse_line(entry["total"])
        checks.append((
            matchup, hcap, fair_handicap(margin),
            tline, fair_total(sim.total), us_total,
        ))

        for laying, who, sign in ((True, favourite, "-"), (False, underdog, "+")):
            candidates.append({
                "matchup": matchup, "market": "讓球",
                "side": f"{who} {sign}{hcap.effective:g}", "line": str(hcap),
                "price": hk_h,
                "raw": handicap_ev(margin, hcap, hk_price=hk_h, laying=laying),
                "adj": handicap_ev(margin, hcap, hk_price=hk_h,
                                   laying=laying, bias=COVER_BIAS),
            })
        # A board does not always quote both sides of a total at the same
        # price. When it splits them the margin sits on one side only, and
        # assuming the common price silently misprices that side.
        prices = {
            True: entry.get("total_price_over", hk_t),
            False: entry.get("total_price_under", hk_t),
        }
        for is_over, label in ((True, "大"), (False, "小")):
            price = prices[is_over]
            ev = total_ev(sim.total, tline, hk_price=price, over=is_over)
            candidates.append({
                "matchup": matchup, "market": "大小",
                "side": f"{label} {tline.effective:g}", "line": str(tline),
                "price": price, "raw": ev, "adj": ev,
            })

    print("=== 解碼驗證：板面線 vs 本場自身隱含的公平線 ===")
    print(f"{'比賽':36s} {'讓球':12s} {'公平':6s} {'差':6s} "
          f"{'大小':12s} {'公平':6s} {'差':6s} {'美盤':5s}")
    h_err, t_err = [], []
    for matchup, hcap, fair_h, tline, fair_t, us_total in checks:
        dh, dt = hcap.effective - fair_h, tline.effective - fair_t
        h_err.append(abs(dh))
        t_err.append(abs(dt))
        print(f"{matchup[:35]:36s} {str(hcap):12s} {fair_h:6.2f} {dh:+6.2f} "
              f"{str(tline):12s} {fair_t:6.2f} {dt:+6.2f} {us_total:5.1f}")
    print(f"\n平均絕對誤差  讓球 {statistics.mean(h_err):.3f} 分   "
          f"大小 {statistics.mean(t_err):.3f} 分")

    evs = [c["adj"] for c in candidates]
    print(f"\n健全性檢查：去偏後 EV 的分布")
    print(f"  中位數 {statistics.median(evs) * 100:+.2f}%  "
          f"(公平掛盤應落在 -2.5% ~ -3.0%)")
    print(f"  範圍 {min(evs) * 100:+.2f}% ~ {max(evs) * 100:+.2f}%")

    candidates.sort(key=lambda c: -c["adj"])
    print(f"\n=== 前 {args.top} 候選（板面實際賠率，去偏後）===")
    print(f"{'#':3s} {'比賽':32s} {'市場':5s} {'選擇':30s} "
          f"{'板面線':13s} {'賠率':6s} {'原始':8s} {'去偏':8s}")
    for i, c in enumerate(candidates[:args.top], 1):
        print(f"{i:<3d} {c['matchup'][:31]:32s} {c['market']:5s} "
              f"{c['side'][:29]:30s} {c['line']:13s} {c['price']:.3f} "
              f"{c['raw'] * 100:+7.2f}% {c['adj'] * 100:+7.2f}%")

    passing = [c for c in candidates if c["adj"] >= 0.04]
    print(f"\n達 +4% 門檻: {len(passing)} / {len(candidates)}")
    if not passing:
        print("不下注 — NO BET")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
