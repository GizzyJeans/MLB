#!/usr/bin/env python3
"""Gate 2B: price board handicaps from US prices alone (docs/gate_2b.md).

  python3 scripts/gate2b.py --board data/boards/<file>.json \
      --snapshot data/snapshots/<file>.json --date YYYY-MM-DD \
      --report reports/<file>.txt [--record]

Every handicap ticket on the board is priced twice: by the model, read from
the pricing report's A ranking, and by the US moneyline and -1.5 run line
alone. A selection passes gate 2B when the model's de-biased EV clears +4%
(gate 1, unchanged) and the US market confirms the edge under the declared
conditions. Gate 3 -- official lineups, then a re-check at lineup time --
still applies to anything that passes.

--record writes the day's result to data/gate2b_ledger.json, which
scripts/track_record.py settles as its own bucket. Re-running for the same
date (a second reading of the board) replaces that date's entry.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlbline.asian import parse_line  # noqa: E402
from mlbline.odds import load_snapshot, normalise, select_slate  # noqa: E402
from mlbline.teams import pad, zh, zh_matchup  # noqa: E402
from mlbline.verify import (  # noqa: E402
    BOOK_AGREEMENT, MARKET_EV_MIN, MIN_BOOKS, check_handicap, kelly)

GATE_1 = 0.04           # model de-biased EV, unchanged
KELLY_SCALE = 0.25      # standing staking rule
STAKE_CAP = 0.01        # single selection, fraction of bankroll
LEDGER = ROOT / "data" / "gate2b_ledger.json"

# Same layout track_record.py parses: rank, game, market, side, line, price,
# raw EV, de-biased EV.
PICK = re.compile(
    r"^(\d+)\s+(.+?)\s+(讓球|大小)\s+(.+?)\s+\S+\([\d.]+\)\s+"
    r"[\d.]+\s+([+-][\d.]+)%\s+([+-][\d.]+)%\s*$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--board", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--date", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--record", action="store_true")
    return p.parse_args()


def model_evs(report: Path) -> dict[tuple[str, str], tuple[float, float]]:
    """(raw, de-biased) EV of each handicap side in the report's A ranking."""
    out, seen = {}, False
    for line in report.read_text("utf-8").splitlines():
        if line.startswith("=== "):
            seen = line.startswith("=== A組")
            continue
        if not (seen and line[:1].isdigit()):
            continue
        match = PICK.match(line)
        if match and match.group(3) == "讓球":
            out[(match.group(2).strip(), match.group(4).strip())] = (
                float(match.group(5)) / 100.0, float(match.group(6)) / 100.0)
    return out


def main() -> int:
    args = parse_args()
    board_path = Path(args.board)
    board = json.loads(board_path.read_text(encoding="utf-8"))
    hk_h = board["handicap_price_hk"]
    games = select_slate(normalise(load_snapshot(args.snapshot)), args.date)
    model = model_evs(Path(args.report))
    floor = min((adj for _, adj in model.values()), default=None)

    print("=== 門檻 2B：只用美盤獨贏 + −1.5 讓分盤替板面讓分定價 ===")
    print(f"通過條件：模型去偏 ≥ +{GATE_1 * 100:.0f}%；市場 EV 四種去水法皆 ≥ "
          f"+{MARKET_EV_MIN * 100:.0f}%；兩條美盤線各 ≥ {MIN_BOOKS} 家；"
          f"≥ {BOOK_AGREEMENT * 100:.0f}% 書商各自 EV > 0；每腳 ≤ 1.5；無讀法衝突")
    print(f"{pad('比賽', 20)} {pad('選擇', 16)} {'模型去偏':>8s} "
          f"{'市場(Shin)':>10s} {'市場最差':>9s} {'書商':>7s} {'同意':>5s} "
          f"{'最低賠率':>8s}  判定")

    passed, hidden = [], 0
    for entry in board["games"]:
        matchup_en = f"{entry['away']} @ {entry['home']}"
        key = (f"{matchup_en} G{entry['game']}" if entry.get("game")
               else matchup_en)
        game = games.get(key)
        if game is None:
            continue
        shown = zh_matchup(entry["away"], entry["home"])
        if entry.get("game"):
            shown += f" G{entry['game']}"
        favourite = entry["favourite"] or game.home_team
        underdog = (game.away_team if favourite == game.home_team
                    else game.home_team)
        hcap = parse_line(entry["handicap"])

        for laying, who, sign in ((True, favourite, "-"),
                                  (False, underdog, "+")):
            side = f"{zh(who)} {sign}{hcap.effective:g}"
            check = check_handicap(game, favourite, hcap, laying, hk_h,
                                   entry.get("reading_conflict", False))
            evs = model.get((shown, side))
            adj = evs[1] if evs else None
            fails = list(check.failures)
            if adj is None:
                # Off the A ranking means below its tenth line.
                if floor is not None and floor >= GATE_1:
                    fails.insert(0, "不在 A 組前 10，門檻 1 無法判定")
                else:
                    fails.insert(0, "門檻 1 未過")
            elif adj < GATE_1:
                fails.insert(0, "門檻 1 未過")

            shin = check.ev.get("shin")
            if max(adj or -1.0, shin if shin is not None else -1.0) <= 0.0:
                hidden += 1
                continue
            adj_txt = f"{adj * 100:+7.2f}%" if adj is not None else "      -"
            shin_txt = f"{shin * 100:+9.2f}%" if shin is not None else "        -"
            worst_txt = (f"{check.ev_min * 100:+8.2f}%" if check.ev
                         else "       -")
            books = f"{check.n_ml}/{check.n_rl}" if check.eligible else "-"
            agree = (f"{check.agreement * 100:4.0f}%" if check.book_evs
                     else "    -")
            price = (f"{check.min_price:8.3f}" if check.min_price
                     else "       -")
            verdict = "通過" if not fails else "；".join(fails)
            print(f"{pad(shown, 20)} {pad(side, 16)} {adj_txt} {shin_txt} "
                  f"{worst_txt} {books:>7s} {agree} {price}  {verdict}")
            if not fails:
                method = check.worst_method
                win, two = check.win[method], check.two_plus[method]
                full = kelly(hcap, laying, hk_h, win, two)
                passed.append({
                    "matchup": shown, "matchup_en": matchup_en,
                    "side": side, "side_en": f"{who} {sign}{hcap.effective:g}",
                    "line": entry["handicap"], "price": hk_h,
                    "model_adj": adj, "market_ev": check.ev,
                    "market_ev_min": check.ev_min, "worst_method": method,
                    "win": win, "two_plus": two,
                    "n_ml": check.n_ml, "n_rl": check.n_rl,
                    "agreement": check.agreement,
                    "min_price": check.min_price,
                    "kelly": full,
                    "stake": min(KELLY_SCALE * full, STAKE_CAP),
                })
    if hidden:
        print(f"（另有 {hidden} 個讓分選擇在模型與市場兩邊都不是正 EV，未列出）")

    print(f"\n通過門檻 2B：{len(passed)} 筆")
    for p in passed:
        one = p["win"] - p["two_plus"]
        print(f"\n  {p['matchup']}  {p['side']}  板面 {p['line']} @ {p['price']:.3f}")
        print(f"    美盤（{p['worst_method']}，最嚴的去水法）：熱門贏 {p['win']:.4f}"
              f"  贏 2 分以上 {p['two_plus']:.4f}  剛好贏 1 分 {one:.4f}")
        print("    市場 EV：" + "  ".join(
            f"{m} {v * 100:+.2f}%" for m, v in p["market_ev"].items()))
        print(f"    模型去偏 EV {p['model_adj'] * 100:+.2f}%   書商 "
              f"{p['n_ml']}/{p['n_rl']}，{p['agreement'] * 100:.0f}% 各自 EV > 0")
        print(f"    最低可接受賠率 {p['min_price']:.3f}   Kelly {p['kelly'] * 100:.1f}%"
              f" → 0.25 Kelly {KELLY_SCALE * p['kelly'] * 100:.2f}%"
              f" → 注碼 {p['stake'] * 100:.2f}%（單注上限 {STAKE_CAP * 100:.0f}%）")
        print("    門檻 3 未完成前只是觀察：打序公布後重抓美盤重驗，"
              "且板面賠率需 ≥ 最低可接受賠率")

    if args.record:
        ledger = (json.loads(LEDGER.read_text(encoding="utf-8"))
                  if LEDGER.exists() else {})
        ledger[args.date] = {
            "board": board_path.stem,
            "snapshot": args.snapshot,
            "report": Path(args.report).stem,
            "recorded_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds"),
            "passed": passed,
        }
        LEDGER.write_text(json.dumps(dict(sorted(ledger.items())),
                                     ensure_ascii=False, indent=1) + "\n",
                          encoding="utf-8")
        print(f"\n已寫入 {LEDGER.relative_to(ROOT)}（{args.date}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
