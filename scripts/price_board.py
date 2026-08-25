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
from mlbline.odds import load_snapshot, normalise, select_slate  # noqa: E402
from mlbline.teams import pad, zh, zh_matchup  # noqa: E402

# Measured shortfall in the favourite's cover probability, from the slate's
# mean disagreement with the market and consistent with the out-of-sample
# check against real games.
COVER_BIAS = 0.0128


def comparable_handicap(level: float) -> float:
    """A handicap level as a bet rather than as a number.

    Baseball has no ties, so every line from -0.5 to +0.5 is the same wager
    -- all of them need an outright win, and all of them price identically.
    A board hanging a pick'em at a flat 0 is therefore exactly on a fair
    line of 0.5, not half a run away from it. Differencing the literal
    numbers reports a 0.5-run decode error that does not exist, and one
    pick'em is enough to double the slate's mean handicap error.

    This also has to be applied to the fair line before it is displayed,
    not only before it is differenced. `fair_handicap` solves for a 50%
    cover, and in a near-coin-flip game no real line delivers that: cover
    sits at 0.4952 flat across the whole [-0.5, +0.5] range and then jumps
    to 0.6046 at -1.0, so the solver interpolates across the step and
    returns something like -0.53 -- a number no line corresponds to. On
    2026-08-25 that put three fair handicaps at about -0.55 for teams the
    market had at 49-50%, next to a difference column reading +0.07, and
    the two could not be reconciled by anyone reading the row.
    """
    return max(level, 0.5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--date", default=None,
                        help="slate date in US Eastern, e.g. 2026-08-14. "
                             "Required once the next day's lines are open.")
    args = parser.parse_args()

    board = json.loads(Path(args.board).read_text(encoding="utf-8"))
    hk_h, hk_t = board["handicap_price_hk"], board["total_price_hk"]
    games = select_slate(normalise(load_snapshot(args.snapshot)), args.date)

    candidates, checks = [], []

    for entry in board["games"]:
        matchup = f"{entry['away']} @ {entry['home']}"
        # A doubleheader is keyed with its game number; a board entry that
        # omits one for such a pairing would otherwise silently price nothing.
        key = f"{matchup} G{entry['game']}" if entry.get("game") else matchup
        game = games.get(key)
        if game is None:
            if any(k.startswith(matchup + " G") for k in games):
                raise SystemExit(
                    f"{matchup} is a doubleheader on this slate; the board "
                    "entry needs a \"game\": 1 or 2.")
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
        # Fit against every total the market quotes, weighted by how many
        # books stand behind each. Anchoring on the modal line alone lets the
        # fit jump when books migrate between lines without the market moving.
        anchors = []
        for point in game.offered_points("totals"):
            summary = summarise_line(game, "totals", point)
            if not summary:
                continue
            side = next(s for s in summary.sides if s.side.startswith("Over"))
            anchors.append((point, side.loo_consensus_prob, float(side.n_books)))
        priced = solve(matchup, market_home_win=home_win, market_over=over,
                       total_line=us_total, over_anchors=anchors)
        sim = priced.simulation

        favourite = entry["favourite"] or game.home_team
        fav_home = favourite == game.home_team
        margin = sim.margin if fav_home else -sim.margin
        underdog = game.away_team if fav_home else game.home_team

        hcap, tline = parse_line(entry["handicap"]), parse_line(entry["total"])
        shown = zh_matchup(entry["away"], entry["home"])
        if entry.get("game"):
            shown += f" G{entry['game']}"
        checks.append((
            shown, hcap, fair_handicap(margin),
            tline, fair_total(sim.total), us_total,
            entry.get("half_total"), priced.anchor_spread,
        ))

        for laying, who, sign in ((True, favourite, "-"), (False, underdog, "+")):
            candidates.append({
                "matchup": shown, "market": "讓球",
                "side": f"{zh(who)} {sign}{hcap.effective:g}", "line": str(hcap),
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
                "matchup": shown, "market": "大小",
                "side": f"{label} {tline.effective:g}", "line": str(tline),
                "price": price, "raw": ev, "adj": ev,
            })

    print("=== 解碼驗證：板面線 vs 本場自身隱含的公平線 ===")
    print(f"{pad('比賽', 24)} {pad('讓球', 14)} {'公平':>6s} {'差':>7s} "
          f"{pad('大小', 14)} {'公平':>6s} {'差':>7s} {'美盤':>5s} {'殘差':>6s}")
    h_err, t_err = [], []
    for matchup, hcap, fair_h, tline, fair_t, us_total, _, spread in checks:
        shown_fair = comparable_handicap(fair_h)
        dh = comparable_handicap(hcap.effective) - shown_fair
        dt = tline.effective - fair_t
        h_err.append(abs(dh))
        t_err.append(abs(dt))
        print(f"{pad(matchup, 24)} {pad(str(hcap), 14)} {shown_fair:6.2f} {dh:+7.2f} "
              f"{pad(str(tline), 14)} {fair_t:6.2f} {dt:+7.2f} {us_total:5.1f} "
              f"{spread * 100:6.2f}")
    print(f"\n平均絕對誤差  讓球 {statistics.mean(h_err):.3f} 分   "
          f"大小 {statistics.mean(t_err):.3f} 分")
    print("末欄為擬合殘差（百分點）：市場自己報的各條大小線彼此是否一致。"
          "數字大代表這場的線互相矛盾，任何外推價都比它的 EV 看起來更不可靠。")

    # Independent check on the transcription, using only the board's own two
    # numbers. Five of nine innings score a fairly stable share of a game's
    # runs, so the first-half total divided by the full-game total should sit
    # in a tight band across a slate. Comparing each game against the slate's
    # own median needs no outside reference and no assumption about what the
    # true share is -- it only asks whether one game was read differently
    # from the rest, which is exactly what a misread digit looks like.
    halves = [(m, t.effective, parse_line(h).effective)
              for m, _, _, t, _, _, h, _ in checks if h]
    if len(halves) >= 4:
        ratios = [half / full for _, full, half in halves]
        mid = statistics.median(ratios)
        print(f"\n上半場一致性檢查（板面自身，不用外部資料）   "
              f"中位比值 {mid:.3f}")
        flagged = [(m, r) for (m, _, _), r in zip(halves, ratios)
                   if abs(r - mid) > 0.05]
        for matchup, full, half in halves:
            r = half / full
            mark = "  ←偏離" if abs(r - mid) > 0.05 else ""
            print(f"  {pad(matchup, 24)} 全場 {full:5.2f}  上半 {half:4.2f}  "
                  f"比值 {r:.3f}{mark}")
        if flagged:
            print(f"  {len(flagged)} 場偏離中位數 0.05 以上 — 先確認抄錄無誤")
        else:
            print("  全部落在中位數 ±0.05 內，抄錄與解碼互相印證")

    evs = [c["adj"] for c in candidates]
    print(f"\n健全性檢查：去偏後 EV 的分布")
    print(f"  中位數 {statistics.median(evs) * 100:+.2f}%  "
          f"(公平掛盤應落在 -2.5% ~ -3.0%)")
    print(f"  範圍 {min(evs) * 100:+.2f}% ~ {max(evs) * 100:+.2f}%")

    candidates.sort(key=lambda c: -c["adj"])
    print(f"\n=== 前 {args.top} 候選（板面實際賠率，去偏後）===")
    print(f"{'#':3s} {pad('比賽', 22)} {pad('市場', 6)} {pad('選擇', 22)} "
          f"{pad('板面線', 14)} {'賠率':>6s} {'原始':>8s} {'去偏':>8s}")
    for i, c in enumerate(candidates[:args.top], 1):
        print(f"{i:<3d} {pad(c['matchup'], 22)} {pad(c['market'], 6)} "
              f"{pad(c['side'], 22)} {pad(c['line'], 14)} {c['price']:.3f} "
              f"{c['raw'] * 100:+7.2f}% {c['adj'] * 100:+7.2f}%")

    passing = [c for c in candidates if c["adj"] >= 0.04]
    print(f"\n達 +4% 門檻: {len(passing)} / {len(candidates)}")
    if not passing:
        print("不下注 — NO BET")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
