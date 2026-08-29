#!/usr/bin/env python3
"""Accumulate every settled slate into one record and test it for skill.

  python3 scripts/track_record.py

Each slate has been settled on its own day, but a day of ten selections
cannot distinguish a real edge from noise -- the standard deviation of a
single Asian bet is close to 100%, so ten bets carry a standard error of
about 30 percentage points against an edge measured in single digits. The
only way the record says anything is pooled.

Two questions are asked of the pool.

The first is whether the ranking has any skill: does a selection the model
liked more actually return more? That is the correlation between predicted
EV and realised return across every settled selection. Its critical value
at 95% confidence is roughly 2/sqrt(n), which for a hundred bets is 0.20 --
so anything inside that band is indistinguishable from a coin.

The second is whether the *level* is right. Both sides of every board line
were also priced, and betting both sides must return exactly minus the
hold. If the pooled both-sides figure drifts away from -2.5% to -3.1%, the
settlement logic disagrees with the pricing logic and one of them is wrong.

The record is written to data/record.csv so the pairs survive, rather than
being re-derived from reports each time.
"""

from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mlbline.asian import handicap_ev, parse_line, total_ev  # noqa: E402
from mlbline.teams import pad, zh, zh_matchup  # noqa: E402
from settle_board import load_scores  # noqa: E402

# date -> (board stem, pricing report stem). The 14th was re-priced against a
# later snapshot after starters moved; the v2 board is the one that stood.
SLATES = [
    ("2026-08-09", "2026-08-09_asian_board", "2026-08-09_board_pricing"),
    ("2026-08-10", "2026-08-10_asian_board", "2026-08-10_board_pricing"),
    ("2026-08-11", "2026-08-11_asian_board", "2026-08-11_board_pricing"),
    ("2026-08-12", "2026-08-12_asian_board", "2026-08-12_board_pricing"),
    ("2026-08-13", "2026-08-13_asian_board", "2026-08-13_board_pricing"),
    ("2026-08-14", "2026-08-14_asian_board_v2", "2026-08-14_board_pricing_v2"),
    ("2026-08-15", "2026-08-15_asian_board", "2026-08-15_board_pricing"),
    ("2026-08-16", "2026-08-16_asian_board", "2026-08-16_board_pricing"),
    ("2026-08-17", "2026-08-17_asian_board", "2026-08-17_board_pricing"),
    ("2026-08-18", "2026-08-18_asian_board", "2026-08-18_board_pricing"),
    ("2026-08-19", "2026-08-19_asian_board", "2026-08-19_board_pricing"),
    ("2026-08-20", "2026-08-20_asian_board_v2", "2026-08-20_board_pricing_v3"),
    ("2026-08-21", "2026-08-21_asian_board", "2026-08-21_board_pricing"),
    ("2026-08-22", "2026-08-22_asian_board", "2026-08-22_board_pricing"),
    ("2026-08-23", "2026-08-23_asian_board", "2026-08-23_board_pricing"),
    ("2026-08-24", "2026-08-24_asian_board", "2026-08-24_board_pricing"),
    ("2026-08-25", "2026-08-25_asian_board", "2026-08-25_board_pricing"),
    ("2026-08-26", "2026-08-26_asian_board", "2026-08-26_board_pricing"),
    ("2026-08-27", "2026-08-27_asian_board", "2026-08-27_board_pricing"),
    ("2026-08-28", "2026-08-28_asian_board", "2026-08-28_board_pricing"),
]

# Boards that were priced once, then moved before first pitch and re-priced.
# The later reading is the day's record above, because it is the board that
# was actually standing when the slate started. These earlier readings are
# settled separately: they are the only place the policy's +4% gate has ever
# fired, so burying them in a superseded report would hide the only evidence
# that exists about it.
EARLY_BOARDS = [
    ("2026-08-14", "2026-08-14_asian_board", "2026-08-14_board_pricing"),
    ("2026-08-20", "2026-08-20_asian_board", "2026-08-20_board_pricing_v2"),
]

# Prospective test declared in docs/prospective_test.md on 2026-08-29, before
# any game in the window had started. Thirty slates, paired daily difference,
# no significance claim before the thirtieth -- the protocol is fixed and the
# stopping rule is not conditional on what the numbers do.
TEST_START = "2026-08-29"
TEST_SLATES = 30
# Declared endpoint for the primary statistic, fixed before any data.
TEST_PAIRS = 2000

# Both EV columns are captured. The raw figure is what the simulation
# priced; the de-biased one adds the measured cover-probability correction,
# which only touches handicaps and only in proportion to each leg's
# exposure. Keeping both lets the correction itself be scored, rather than
# only the number it produced.
PICK = re.compile(
    r"^(\d+)\s+(.+?)\s+(讓球|大小)\s+(.+?)\s+\S+\([\d.]+\)\s+"
    r"[\d.]+\s+([+-][\d.]+)%\s+([+-][\d.]+)%\s*$")


def settle_slate(date: str, board_stem: str):
    """Every priced selection on one slate, with its realised return."""
    board = json.loads(
        (ROOT / "data" / "boards" / f"{board_stem}.json").read_text("utf-8"))
    hk_h, hk_t = board["handicap_price_hk"], board["total_price_hk"]
    scores = load_scores(
        str(ROOT / "data" / "scores" / f"aug{date[-2:]}_scores.json"), date)

    rows = []
    for entry in board["games"]:
        matchup_en = f"{entry['away']} @ {entry['home']}"
        key = (f"{matchup_en} G{entry['game']}" if entry.get("game")
               else matchup_en)
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
        matchup = zh_matchup(entry["away"], entry["home"])
        if entry.get("game"):
            matchup += f" G{entry['game']}"

        for laying, who, sign in ((True, favourite, "-"),
                                  (False, underdog, "+")):
            rows.append({
                "date": date, "matchup": matchup, "matchup_en": matchup_en,
                "market": "讓球",
                "side": f"{zh(who)} {sign}{hcap.effective:g}",
                "side_en": f"{who} {sign}{hcap.effective:g}",
                "price": hk_h,
                "kind": "讓分" if laying else "受讓",
                "pnl": handicap_ev(margin, hcap, hk_price=hk_h, laying=laying),
            })
        prices = {True: entry.get("total_price_over", hk_t),
                  False: entry.get("total_price_under", hk_t)}
        for is_over, name in ((True, "大"), (False, "小")):
            rows.append({
                "date": date, "matchup": matchup, "matchup_en": matchup_en,
                "market": "大小",
                "side": f"{name} {tline.effective:g}",
                "side_en": f"{name} {tline.effective:g}",
                "price": prices[is_over],
                "kind": "大分" if is_over else "小分",
                "pnl": total_ev(total, tline, hk_price=prices[is_over],
                                over=is_over),
            })
    return rows


def top_picks(report_stem: str, rows: list[dict], limit: int = 10,
              arm: str = "A"):
    """The report's ranked selections, paired with what they returned.

    Reports written before 2026-08-29 carry a single ranking under a
    "=== 前 N 候選" heading. From the 29th they carry two, "A組" (ranked on
    the de-biased EV, the live one) and "B組" (the same candidates ranked on
    the raw EV), for the prospective test declared in
    docs/prospective_test.md. Asking for arm "B" on an older report returns
    nothing rather than silently handing back the A ranking.
    """
    text = (ROOT / "reports" / f"{report_stem}.txt").read_text("utf-8")
    out, seen = [], False
    for line in text.splitlines():
        if line.startswith("=== "):
            if line.startswith("=== 前"):
                seen = arm == "A"          # legacy single-ranking report
            elif line.startswith(f"=== {arm}組"):
                seen = True
            else:
                seen = False
            continue
        if not (seen and line and line[0].isdigit()):
            continue
        match = PICK.match(line)
        if not match:
            continue
        rank, want_game, market, side_text, raw, adj = match.groups()
        want_side = side_text.strip()[:12]
        row = next(
            (r for r in rows
             if r["market"] == market
             and (r["matchup"].startswith(want_game.strip())
                  or r["matchup_en"].startswith(want_game.strip()))
             and (r["side"].startswith(want_side)
                  or r["side_en"].startswith(want_side))),
            None)
        if row is None:
            continue
        out.append({**row, "rank": int(rank),
                    "raw_ev": float(raw) / 100.0,
                    "expected": float(adj) / 100.0})
        if len(out) == limit:
            break
    return out


def main() -> int:
    picks, everything = [], []
    print(f"{pad('日期', 12)} {'注數':>4s} {'贏':>3s} {'輸':>3s} "
          f"{'預期':>8s} {'實際':>10s} {pad('全板兩面', 10)}")
    for date, board_stem, report_stem in SLATES:
        rows = settle_slate(date, board_stem)
        if not rows:
            print(f"{pad(date, 12)}  (無結果)")
            continue
        chosen = top_picks(report_stem, rows)
        everything.extend(rows)
        picks.extend(chosen)
        pnl = [c["pnl"] for c in chosen]
        both = sum(r["pnl"] for r in rows) / len(rows)
        print(f"{pad(date, 12)} {len(chosen):4d} "
              f"{sum(1 for p in pnl if p > 0.01):3d} "
              f"{sum(1 for p in pnl if p < -0.01):3d} "
              f"{statistics.mean(c['expected'] for c in chosen) * 100:+7.2f}% "
              f"{statistics.mean(pnl) * 100:+9.2f}% "
              f"{both * 100:+9.2f}%")

    with (ROOT / "data" / "record.csv").open("w", newline="",
                                             encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["date", "rank", "matchup_en", "market", "side_en",
                            "kind", "price", "raw_ev", "expected", "pnl"],
            extrasaction="ignore")
        writer.writeheader()
        writer.writerows(picks)

    exp = [c["expected"] for c in picks]
    got = [c["pnl"] for c in picks]
    n = len(picks)
    print(f"\n=== 累計 {n} 筆推薦 ===")
    print(f"  平均預期 EV {statistics.mean(exp) * 100:+.2f}%   "
          f"平均實際 {statistics.mean(got) * 100:+.2f}%")
    print(f"  合計 {sum(got) * 100:+.2f}%   "
          f"每注標準差 {statistics.stdev(got) * 100:.1f}%")
    sem = statistics.stdev(got) / (n ** 0.5)
    print(f"  平均值標準誤 {sem * 100:.2f}%  →  95% 區間 "
          f"{(statistics.mean(got) - 1.96 * sem) * 100:+.2f}% ~ "
          f"{(statistics.mean(got) + 1.96 * sem) * 100:+.2f}%")

    r = float(np.corrcoef(exp, got)[0, 1])
    crit = 1.96 / ((n - 2) ** 0.5)
    print(f"\n  排序技巧 r = {r:+.3f}   臨界值 ±{crit:.3f}   "
          f"{'有訊號' if abs(r) > crit else '無法與雜訊區分'}")

    mid = statistics.median(exp)
    hi = [g for e, g in zip(exp, got) if e >= mid]
    lo = [g for e, g in zip(exp, got) if e < mid]
    print(f"  預期較高的一半 {statistics.mean(hi) * 100:+.2f}%  "
          f"較低的一半 {statistics.mean(lo) * 100:+.2f}%  "
          f"差 {(statistics.mean(hi) - statistics.mean(lo)) * 100:+.2f}pp")

    # A/B arms of the prospective test. Both rankings come from one candidate
    # pool priced once; only the sort key differs.
    ab = []
    for date, board_stem, report_stem in SLATES:
        if date < TEST_START:
            continue
        rows = settle_slate(date, board_stem)
        arms = {k: top_picks(report_stem, rows, arm=k) for k in ("A", "B")}
        if not (arms["A"] and arms["B"]):
            continue
        means = {k: statistics.mean(c["pnl"] for c in v) for k, v in arms.items()}
        overlap = len({(c["matchup"], c["market"], c["side"]) for c in arms["A"]}
                      & {(c["matchup"], c["market"], c["side"]) for c in arms["B"]})
        ab.append((date, means["A"], means["B"], means["B"] - means["A"], overlap))

    print(f"\n=== 前瞻測試 A(去偏) vs B(原始)   第 {len(ab)} / {TEST_SLATES} 天 ===")
    if not ab:
        print(f"  尚未開始（第一個測試日為 {TEST_START} 之後的第一張盤）")
    else:
        print(f"{pad('日期', 12)} {'A組':>9s} {'B組':>9s} {'差(B-A)':>10s} {'重疊':>5s}")
        for date, a, b, d, ov in ab:
            print(f"{pad(date, 12)} {a * 100:+8.2f}% {b * 100:+8.2f}% "
                  f"{d * 100:+9.2f}% {ov:4d}")
        diffs = [d for _, _, _, d, _ in ab]
        print(f"\n  A 組平均 {statistics.mean(x for _, x, _, _, _ in ab) * 100:+.2f}%"
              f"   B 組平均 {statistics.mean(x for _, _, x, _, _ in ab) * 100:+.2f}%"
              f"   平均配對差 {statistics.mean(diffs) * 100:+.2f}%")
        print(f"  平均重疊 {statistics.mean(x for _, _, _, _, x in ab):.1f} / 10")
        if len(diffs) >= 2:
            sd = statistics.stdev(diffs)
            print(f"  配對差標準差 {sd * 100:.1f}pp   "
                  f"標準誤 {sd / len(diffs) ** 0.5 * 100:.1f}pp")
        if len(ab) < TEST_SLATES:
            print(f"  還需 {TEST_SLATES - len(ab)} 天。"
                  "依宣告的停止規則，第 30 天之前不做任何顯著性宣稱。")
        else:
            t = statistics.mean(diffs) / (statistics.stdev(diffs)
                                          / len(diffs) ** 0.5)
            print(f"  t = {t:+.2f}   df = {len(diffs) - 1}   "
                  "（雙尾 α=0.05，臨界值約 ±2.05）")

    # Declared primary statistic: the within-line contrast. The correction
    # adds to the laying side and subtracts from the receiving side, so the
    # difference between the two sides of one handicap is exactly the
    # treatment contrast, with the game itself differenced away.
    from collections import defaultdict
    by_line = defaultdict(dict)
    for r in everything:
        if r["market"] != "讓球":
            continue
        by_line[(r["date"], r["matchup_en"])][r["kind"]] = r["pnl"]
    pairs = [v["讓分"] - v["受讓"] for v in by_line.values() if len(v) == 2]
    print(f"\n=== 主要統計量：線內配對對比（讓分 − 受讓）===")
    print(f"  累積 {len(pairs)} / {TEST_PAIRS} 組")
    if len(pairs) >= 2:
        mean = statistics.mean(pairs)
        sd = statistics.stdev(pairs)
        sem = sd / len(pairs) ** 0.5
        print(f"  平均 {mean * 100:+.2f}pp   sd {sd * 100:.1f}pp   "
              f"標準誤 {sem * 100:.1f}pp   t = {mean / sem:+.2f}")
        print(f"  目前可偵測（80% 檢定力）{2.8 * sem * 100:.1f}pp")
    if len(pairs) < TEST_PAIRS:
        print(f"  尚未達終點，依宣告不做顯著性宣稱。"
              f"還需 {TEST_PAIRS - len(pairs)} 組。")

    allp = [r["pnl"] for r in everything]
    print(f"\n  健全性：全部 {len(allp)} 個板面選擇（兩面全下）"
          f"{statistics.mean(allp) * 100:+.2f}% 每注")
    print("  （應等於 -抽水 ≈ -2.6% 讓球 / -3.1% 大小）")

    # The gate is the whole policy. Everything above is what the ranking did
    # on selections the policy declined, which answers a different question.
    gated = [c for c in picks if c["expected"] >= 0.04]
    for date, board_stem, report_stem in EARLY_BOARDS:
        rows = settle_slate(date, board_stem)
        if rows:
            gated.extend(c for c in top_picks(report_stem, rows)
                         if c["expected"] >= 0.04)
    print(f"\n=== 通過 +4% 門檻的選擇（含移動前的早盤）: {len(gated)} 筆 ===")
    if not gated:
        print("  無")
        return 0
    for c in sorted(gated, key=lambda c: -c["expected"]):
        verdict = ("贏" if c["pnl"] > 0.01
                   else ("輸" if c["pnl"] < -0.01 else "和"))
        print(f"  {c['date']} {pad(c['matchup'], 20)} {pad(c['side'], 16)} "
              f"{c['expected'] * 100:+6.2f}% → {c['pnl'] * 100:+7.2f}%  {verdict}")
    got = [c["pnl"] for c in gated]
    n = len(got)
    print(f"\n  合計 {sum(got) * 100:+.2f}%   平均 {statistics.mean(got) * 100:+.2f}% 每注")
    print(f"  平均預期 {statistics.mean(c['expected'] for c in gated) * 100:+.2f}%")
    if n >= 2:
        sem = statistics.stdev(got) / (n ** 0.5)
        print(f"  標準誤 {sem * 100:.1f}%  →  95% 區間 "
              f"{(statistics.mean(got) - 1.96 * sem) * 100:+.1f}% ~ "
              f"{(statistics.mean(got) + 1.96 * sem) * 100:+.1f}%")
    wins = sum(1 for p in got if p > 0.01)
    print(f"  {wins} 勝 {sum(1 for p in got if p < -0.01)} 敗 "
          f"{n - wins - sum(1 for p in got if p < -0.01)} 和")
    print(f"  在真實勝率 53% 的假設下，{n} 戰全勝的機率約 "
          f"{0.53 ** n * 100:.0f}% — 樣本量還不足以說明任何事")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
