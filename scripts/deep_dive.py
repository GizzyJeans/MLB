#!/usr/bin/env python3
"""Decompose where each board candidate's EV actually comes from.

  python3 scripts/deep_dive.py --board data/boards/<file>.json \
      --snapshot data/snapshots/<file>.json --date YYYY-MM-DD

The daily report answers "what is the EV". This answers "why", which is a
different and more useful question when the EV column is the only thing
standing between a number and a bet.

Every candidate's edge is the gap between the board's line and the model's
fair line. That gap decomposes into two pieces that look identical in an EV
column and are not remotely equally trustworthy:

  board - market   the Asian book hangs a different number than fifteen US
                   books do. This needs no opinion from the model at all --
                   it is one price against another, and the model is only
                   being asked to convert a line difference into a
                   probability.

  market - fair    the model disagrees with the market itself. This is the
                   model claiming to know something fifteen books do not,
                   and it is only as good as the simulation's calibration.

A candidate driven by the first is a line-shopping edge. One driven by the
second is a forecasting claim. The record has one clean instance: on 09-14
the two top candidates split exactly this way, the board-driven one won and
the model-driven one lost. That is a single observation and proves nothing,
but the distinction is worth making before the fact rather than after.

Two further robustness columns:

  sensitivity      how far the EV moves when the line is nudged by the
                   slate's own typical decode error. Moving the line up is
                   the same as moving the fair value down, so this measures
                   the gap's robustness from either side. An EV that
                   survives the nudge is a different object from one that
                   does not.

  dispersion       how much the fifteen books disagree among themselves,
                   measured as the spread across de-vigging methods and the
                   book count behind the line. A market that cannot agree
                   with itself is a weak reference to be measuring against.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlbline.asian import (  # noqa: E402
    fair_total as solve_fair_total, handicap_ev, parse_line, total_ev)
# Building a line from an arbitrary float is exactly what the fair-line
# solvers already do internally. parse_line only accepts board notation and
# clean halves, so a nudged 1.05 is not expressible through it.
from mlbline.asian import _as_line  # noqa: E402
from mlbline.implied import solve  # noqa: E402
from mlbline.market import summarise_line  # noqa: E402
from mlbline.odds import load_snapshot, normalise, select_slate  # noqa: E402
from mlbline.teams import pad, zh, zh_matchup  # noqa: E402

# Nudge applied to the fitted fair line when testing robustness. This is the
# slate's own mean absolute decode error, rounded: the amount by which the
# board and the model routinely differ for reasons that are not edges.
NUDGE = 0.15


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--board", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--date", required=True)
    return p.parse_args()


def fit(game):
    """Fit the scoring model to one game's market, as price_board does."""
    us_total = game.modal_point("totals")
    h2h = summarise_line(game, "h2h", None)
    totals = summarise_line(game, "totals", us_total)
    if not (h2h and totals):
        return None
    home = next(s for s in h2h.sides if s.side == game.home_team)
    over = next(s for s in totals.sides if s.side.startswith("Over"))
    anchors = []
    for point in game.offered_points("totals"):
        summary = summarise_line(game, "totals", point)
        if not summary:
            continue
        side = next(s for s in summary.sides if s.side.startswith("Over"))
        anchors.append((point, side.loo_consensus_prob, float(side.n_books)))
    priced = solve(game.matchup, market_home_win=home.loo_consensus_prob,
                   market_over=over.loo_consensus_prob, total_line=us_total,
                   over_anchors=anchors)
    return us_total, h2h, totals, home, over, priced


def main() -> int:
    args = parse_args()
    board = json.loads(Path(args.board).read_text(encoding="utf-8"))
    hk_h, hk_t = board["handicap_price_hk"], board["total_price_hk"]
    games = select_slate(normalise(load_snapshot(args.snapshot)), args.date)

    rows = []
    for entry in board["games"]:
        matchup_en = f"{entry['away']} @ {entry['home']}"
        key = (f"{matchup_en} G{entry['game']}" if entry.get("game")
               else matchup_en)
        game = games.get(key)
        if game is None:
            continue
        fitted = fit(game)
        if fitted is None:
            print(f"{matchup_en}: 市場不完整，跳過")
            continue
        us_total, h2h, totals, home, over, priced = fitted
        sim = priced.simulation

        favourite = entry["favourite"] or game.home_team
        fav_home = favourite == game.home_team
        margin = sim.margin if fav_home else -sim.margin
        underdog = game.away_team if fav_home else game.home_team

        hcap, tline = parse_line(entry["handicap"]), parse_line(entry["total"])
        fair = solve_fair_total(sim.total)

        rows.append({
            "shown": zh_matchup(entry["away"], entry["home"]),
            "game": game, "entry": entry, "priced": priced, "sim": sim,
            "us_total": us_total, "h2h": h2h, "totals": totals,
            "home_side": home, "over_side": over,
            "hcap": hcap, "tline": tline, "margin": margin,
            "favourite": favourite, "underdog": underdog,
            "fair_total": fair,
            "score": priced.predicted_score,
        })

    # ---------- 1. where the totals edge comes from ----------
    print("=== 大小盤：優勢來自哪裡 ===")
    print("板面 = 亞盤掛的線；美盤 = 15 家書商的主流線；公平 = 模型擬合值")
    print(f"{pad('比賽', 22)} {'板面':>7s} {'美盤':>6s} {'公平':>7s} "
          f"{'板-市':>7s} {'市-公':>7s} {'來源':>10s} {'書商':>5s} {'殘差':>6s}")
    for r in rows:
        board_t = r["tline"].effective
        fair = r["fair_total"]
        bm, mf = board_t - r["us_total"], r["us_total"] - fair
        if abs(bm) >= 2 * abs(mf):
            src = "板面價差"
        elif abs(mf) >= 2 * abs(bm):
            src = "模型分歧"
        else:
            src = "兩者各半"
        print(f"{pad(r['shown'], 22)} {board_t:7.2f} {r['us_total']:6.1f} "
              f"{fair:7.2f} {bm:+7.2f} {mf:+7.2f} {pad(src, 10)} "
              f"{r['over_side'].n_books:5d} {r['priced'].anchor_spread*100:6.2f}")

    # ---------- 2. where the handicap edge comes from ----------
    print("\n=== 讓球盤：優勢來自哪裡 ===")
    print("美盤只掛 ±1.5，所以這裡用『熱門隊獨贏機率』對齊三方")
    print(f"{pad('比賽', 22)} {pad('熱門', 12)} {'板面讓':>7s} "
          f"{'市場勝':>7s} {'模型勝':>7s} {'差':>7s} {'方法散布':>9s}")
    for r in rows:
        g = r["game"]
        fav_home = r["favourite"] == g.home_team
        mkt = (r["home_side"].loo_consensus_prob if fav_home
               else 1 - r["home_side"].loo_consensus_prob)
        mdl = float((r["margin"] > 0).mean())
        print(f"{pad(r['shown'], 22)} {pad(zh(r['favourite']), 12)} "
              f"{r['hcap'].effective:7.3f} {mkt:7.4f} {mdl:7.4f} "
              f"{(mdl-mkt)*100:+6.2f}pp {r['home_side'].method_risk*100:8.2f}pp")

    # ---------- 3. robustness of every positive candidate ----------
    print(f"\n=== 正 EV 候選的穩健度（把線推 ±{NUDGE} 分）===")
    print("EV 欄位不會告訴你這個數字有多脆。把線推一個『常態解碼誤差』的")
    print("距離（等同於把公平值往反方向推同樣多），看 EV 還剩多少。")
    print(f"{pad('比賽', 22)} {pad('選擇', 20)} {'EV':>8s} "
          f"{'線+0.15':>9s} {'線-0.15':>9s} {'最差':>8s} {'撐得住?':>8s}")
    cands = []
    for r in rows:
        for is_over, name in ((True, "大"), (False, "小")):
            # A board can split the two sides of a total. price_board reads
            # the per-side price; this has to match it or the two reports
            # disagree about the same candidate.
            price = r["entry"].get(
                "total_price_over" if is_over else "total_price_under", hk_t)
            ev = total_ev(r["sim"].total, r["tline"], hk_price=price,
                          over=is_over)
            if ev <= 0:
                continue
            lo = _as_line(r["tline"].effective - NUDGE)
            hi = _as_line(r["tline"].effective + NUDGE)
            a = total_ev(r["sim"].total, hi, hk_price=price, over=is_over)
            b = total_ev(r["sim"].total, lo, hk_price=price, over=is_over)
            cands.append((r["shown"], f"{name} {r['tline'].effective:g}",
                          ev, a, b))
        for laying, who in ((True, r["favourite"]), (False, r["underdog"])):
            ev = handicap_ev(r["margin"], r["hcap"], hk_price=hk_h,
                             laying=laying)
            if ev <= 0:
                continue
            sign = "-" if laying else "+"
            lo = _as_line(max(r["hcap"].effective - NUDGE, 0.0))
            hi = _as_line(r["hcap"].effective + NUDGE)
            a = handicap_ev(r["margin"], hi, hk_price=hk_h, laying=laying)
            b = handicap_ev(r["margin"], lo, hk_price=hk_h, laying=laying)
            cands.append((r["shown"],
                          f"{zh(who)} {sign}{r['hcap'].effective:g}",
                          ev, a, b))
    cands.sort(key=lambda c: -c[2])
    for shown, side, ev, a, b in cands:
        worst = min(a, b)
        ok = "是" if worst > 0 else "否"
        print(f"{pad(shown, 22)} {pad(side, 20)} {ev*100:+7.2f}% "
              f"{a*100:+8.2f}% {b*100:+8.2f}% {worst*100:+7.2f}% "
              f"{pad(ok, 8)}")
    if not cands:
        print("  （全盤沒有任何正 EV 候選）")

    # ---------- 4. how trustworthy is the reference market ----------
    print("\n=== 參考市場本身有多可靠 ===")
    print("模型是被市場校準的，所以市場自己越不一致，任何外推價越不可信。")
    print(f"{pad('比賽', 22)} {'獨贏書商':>9s} {'方法散布':>9s} "
          f"{'大小書商':>9s} {'擬合殘差':>9s} {'最佳獨贏報價':>14s}")
    for r in rows:
        g = r["game"]
        best = max(g.select("h2h"), key=lambda q: q.decimal, default=None)
        tag = f"{zh(best.side)} {best.decimal:.2f}@{best.book}" if best else "-"
        print(f"{pad(r['shown'], 22)} {r['home_side'].n_books:9d} "
              f"{r['home_side'].method_risk*100:8.2f}pp "
              f"{r['over_side'].n_books:9d} "
              f"{r['priced'].anchor_spread*100:8.2f}pp {pad(tag, 14)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
