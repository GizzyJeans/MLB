#!/usr/bin/env python3
"""Settle a priced Asian board against final scores.

  python3 scripts/settle_board.py --board data/boards/<file>.json \
      --scores data/scores/<file>.json

Settlement reuses the same pricing functions the forecast used, with the
final score standing in for the distribution. A realised return is just an
expectation over a one-point distribution, so pushes, split tickets and the
handicap's direction are all handled by the code that priced them rather
than by a second implementation that could disagree with the first.

A caution that belongs next to the output: none of these were bets. Every
one failed the staking threshold, so this settles what was explicitly
declined. Ten results also cannot measure an edge of a few percent -- the
noise on ten outcomes swamps it by an order of magnitude. What this catches
is gross error: a line decoded backwards, a handicap settled to the wrong
side, a market whose结果 contradicts the sign of its own price.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlbline.asian import handicap_ev, parse_line, total_ev  # noqa: E402


def load_scores(path: str) -> dict[tuple[str, str], tuple[int, int]]:
    """Map (away, home) -> (away runs, home runs) for completed games."""
    out = {}
    for game in json.loads(Path(path).read_text(encoding="utf-8")):
        if not game.get("completed") or not game.get("scores"):
            continue
        runs = {s["name"]: int(s["score"]) for s in game["scores"]}
        away, home = game["away_team"], game["home_team"]
        if away in runs and home in runs:
            out[(away, home)] = (runs[away], runs[home])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--expected", default=None,
                        help="saved pricing report, to compare against")
    args = parser.parse_args()

    board = json.loads(Path(args.board).read_text(encoding="utf-8"))
    hk_h, hk_t = board["handicap_price_hk"], board["total_price_hk"]
    scores = load_scores(args.scores)

    rows = []
    for entry in board["games"]:
        key = (entry["away"], entry["home"])
        if key not in scores:
            continue
        away_runs, home_runs = scores[key]
        margin_home = np.array([home_runs - away_runs])
        total = np.array([away_runs + home_runs])

        favourite = entry["favourite"] or entry["home"]
        fav_home = favourite == entry["home"]
        margin = margin_home if fav_home else -margin_home
        underdog = entry["away"] if fav_home else entry["home"]

        hcap, tline = parse_line(entry["handicap"]), parse_line(entry["total"])
        label = f"{entry['away'][:3].upper()}@{entry['home'][:3].upper()}"
        result = f"{away_runs}-{home_runs}"

        matchup = f"{entry['away']} @ {entry['home']}"
        for laying, who, sign in ((True, favourite, "-"), (False, underdog, "+")):
            rows.append({
                "game": label, "matchup": matchup, "result": result,
                "market": "讓球", "side": f"{who} {sign}{hcap.effective:g}",
                "pnl": handicap_ev(margin, hcap, hk_price=hk_h, laying=laying),
            })
        for is_over, name in ((True, "大"), (False, "小")):
            rows.append({
                "game": label, "matchup": matchup, "result": result,
                "market": "大小", "side": f"{name} {tline.effective:g}",
                "pnl": total_ev(total, tline, hk_price=hk_t, over=is_over),
            })

    if args.expected:
        text = Path(args.expected).read_text(encoding="utf-8")
        picks, seen = [], False
        for line in text.splitlines():
            if line.startswith("=== 前"):
                seen = True
                continue
            if seen and line and line[0].isdigit():
                picks.append(line)
        print("=== 前 10 候選結算 ===")
        print(f"{'#':3s} {'比賽':12s} {'比分':7s} {'選擇':30s} "
              f"{'預期EV':9s} {'實際':9s} {'結果':6s}")
        realised = []
        # Match on the game as well as the selection. Matching on the
        # selection alone silently picks whichever row sorts first, and two
        # different games can carry the same text -- "小 8" and "大 8" both
        # appear more than once on a slate.
        pattern = re.compile(
            r"^(\d+)\s+(.+?)\s+(讓球|大小)\s+(.+?)\s+\S+\([\d.]+\)\s+"
            r"[\d.]+\s+[+-][\d.]+%\s+([+-][\d.]+)%\s*$")
        for line in picks[:10]:
            match = pattern.match(line)
            if not match:
                print(f"(無法解析) {line[:60]}")
                continue
            rank, matchup_prefix, market, side_text, adj_text = match.groups()
            adj = float(adj_text)
            side = next(
                (r for r in rows
                 if r["matchup"].startswith(matchup_prefix.strip())
                 and r["market"] == market
                 and r["side"].startswith(side_text.strip()[:12])),
                None)
            if side is None:
                print(f"{rank:3s} (無法對應) {matchup_prefix} / {side_text}")
                continue
            pnl = side["pnl"]
            realised.append(pnl)
            verdict = "贏" if pnl > 0.01 else ("輸" if pnl < -0.01 else "和")
            print(f"{rank:3s} {side['game']:12s} {side['result']:7s} "
                  f"{side['side'][:29]:30s} {adj:+8.2f}% {pnl * 100:+8.2f}% "
                  f"{verdict:6s}")
        if realised:
            print(f"\n前 10 合計損益: {sum(realised) * 100:+.2f}%  "
                  f"平均 {statistics.mean(realised) * 100:+.2f}% 每注")

    print(f"\n=== 全部 {len(rows)} 個板面選擇 ===")
    total_pnl = sum(r["pnl"] for r in rows)
    wins = sum(1 for r in rows if r["pnl"] > 0.01)
    losses = sum(1 for r in rows if r["pnl"] < -0.01)
    print(f"  贏 {wins}  輸 {losses}  和 {len(rows) - wins - losses}")
    print(f"  合計 {total_pnl * 100:+.2f}%   平均 "
          f"{total_pnl / len(rows) * 100:+.2f}% 每注")
    print("  （兩邊都下必然約等於 -抽水，這是結算邏輯的健全性檢查）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
