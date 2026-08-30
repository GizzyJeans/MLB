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
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlbline.asian import handicap_ev, parse_line, total_ev  # noqa: E402
from mlbline.teams import pad, zh, zh_matchup  # noqa: E402


def abbrev(team: str) -> str:
    """Short tag that stays distinct across same-city clubs."""
    words = team.split()
    if len(words) >= 2:
        return (words[0][:2] + words[-1][:2]).upper()
    return team[:4].upper()


def eastern_date(iso: str) -> str:
    """Calendar date of a game in US Eastern time.

    A slate spans two UTC dates -- a 10pm Eastern first pitch is already
    tomorrow in UTC -- so the UTC date splits one night's games in two.
    Eastern is what a schedule means by "the 11th".
    """
    stamp = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return stamp.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")


def load_scores(path: str, date: str | None = None
                ) -> dict[str, tuple[int, int]]:
    """Map a slate key to (away runs, home runs) for completed games.

    Keyed the same way `odds.select_slate` keys lines: "<away> @ <home>" for a
    single game, with " G1" / " G2" appended in start order when a pairing is
    played twice on one date. Both sides of the pipeline therefore name a
    doubleheader identically, and a settlement cannot quietly grade one game
    against the other's score.

    `date` is not optional in practice. Teams play series, so the same
    pairing also completes on consecutive days; without the filter a
    settlement grades yesterday's picks against tonight's scores.
    """
    grouped: dict[str, list[tuple[str, tuple[int, int]]]] = {}
    for game in json.loads(Path(path).read_text(encoding="utf-8")):
        if not game.get("completed") or not game.get("scores"):
            continue
        if date and eastern_date(game["commence_time"]) != date:
            continue
        runs = {s["name"]: int(s["score"]) for s in game["scores"]}
        away, home = game["away_team"], game["home_team"]
        if away not in runs or home not in runs:
            continue
        grouped.setdefault(f"{away} @ {home}", []).append(
            (game["commence_time"], (runs[away], runs[home])))

    out: dict[str, tuple[int, int]] = {}
    for matchup, entries in grouped.items():
        entries.sort()
        if len(entries) == 1:
            out[matchup] = entries[0][1]
            continue
        for number, (_, score) in enumerate(entries, start=1):
            out[f"{matchup} G{number}"] = score
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--expected", default=None,
                        help="saved pricing report, to compare against")
    parser.add_argument("--date", required=True,
                        help="slate date in US Eastern, e.g. 2026-08-11")
    args = parser.parse_args()

    board = json.loads(Path(args.board).read_text(encoding="utf-8"))
    hk_h, hk_t = board["handicap_price_hk"], board["total_price_hk"]
    scores = load_scores(args.scores, args.date)

    rows = []
    for entry in board["games"]:
        matchup_en = f"{entry['away']} @ {entry['home']}"
        key = f"{matchup_en} G{entry['game']}" if entry.get("game") else matchup_en
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
        # Three letters collide: San Diego and San Francisco both give
        # "SAN", which reads as the same game on two different rows.
        label = f"{zh(entry['away'])}@{zh(entry['home'])}"
        result = f"{away_runs}-{home_runs}"

        matchup = zh_matchup(entry["away"], entry["home"])
        if entry.get("game"):
            matchup += f" G{entry['game']}"
            label += f" G{entry['game']}"
        for laying, who, sign in ((True, favourite, "-"), (False, underdog, "+")):
            rows.append({
                "game": label, "matchup": matchup, "matchup_en": matchup_en,
                "result": result, "market": "讓球",
                "side": f"{zh(who)} {sign}{hcap.effective:g}",
                "side_en": f"{who} {sign}{hcap.effective:g}",
                "pnl": handicap_ev(margin, hcap, hk_price=hk_h, laying=laying),
            })
        # Honour the same per-side price overrides the pricing step reads.
        # Settling a split-priced total at the common price grades a bet at
        # odds it was never offered at, and the two scripts must not drift.
        prices = {
            True: entry.get("total_price_over", hk_t),
            False: entry.get("total_price_under", hk_t),
        }
        for is_over, name in ((True, "大"), (False, "小")):
            rows.append({
                "game": label, "matchup": matchup, "matchup_en": matchup_en,
                "result": result, "market": "大小",
                "side": f"{name} {tline.effective:g}",
                "side_en": f"{name} {tline.effective:g}",
                "pnl": total_ev(total, tline, hk_price=prices[is_over],
                                over=is_over),
            })

    if args.expected:
        text = Path(args.expected).read_text(encoding="utf-8")
        # Reports written before 2026-08-29 carry one ranking under
        # "=== 前 N 候選"; from the 29th they carry two, "A組" and "B組", for
        # the prospective test. Settle the A arm -- the live one -- and stop
        # at the B heading rather than running the two together.
        picks, seen = [], False
        for line in text.splitlines():
            if line.startswith("=== "):
                seen = (line.startswith("=== 前")
                        or line.startswith("=== A組"))
                continue
            if seen and line and line[0].isdigit():
                picks.append(line)
        # An empty parse is the failure this guard exists for. When the
        # report's heading changed on 2026-08-29 the parser silently matched
        # nothing and printed an empty table -- readable as "no picks today"
        # rather than as a broken parser. A caller who asked for a
        # comparison and got none should be told.
        if not picks:
            raise SystemExit(
                f"{args.expected} 裡找不到任何可解析的候選列。"
                "報告的區塊標題可能已更動。")
        print("=== 前 10 候選結算 ===")
        print(f"{'#':3s} {pad('比賽', 18)} {'比分':>7s} {pad('選擇', 22)} "
              f"{'預期EV':>9s} {'實際':>9s} {'結果':>6s}")
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
            # Reports written before teams were displayed in Chinese carry
            # English names, so match either form rather than stranding them.
            want_game, want_side = matchup_prefix.strip(), side_text.strip()[:12]
            side = next(
                (r for r in rows
                 if r["market"] == market
                 and (r["matchup"].startswith(want_game)
                      or r["matchup_en"].startswith(want_game))
                 and (r["side"].startswith(want_side)
                      or r["side_en"].startswith(want_side))),
                None)
            if side is None:
                print(f"{rank:3s} (無法對應) {matchup_prefix} / {side_text}")
                continue
            pnl = side["pnl"]
            realised.append(pnl)
            verdict = "贏" if pnl > 0.01 else ("輸" if pnl < -0.01 else "和")
            print(f"{rank:3s} {pad(side['game'], 18)} {side['result']:>7s} "
                  f"{pad(side['side'], 22)} {adj:+8.2f}% {pnl * 100:+8.2f}% "
                  f"   {verdict}")
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
