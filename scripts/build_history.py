#!/usr/bin/env python3
"""Extract game results from Retrosheet box score files.

  python3 scripts/build_history.py --retrosheet /path/to/retrosheet \
      --seasons 2023 2024 2025 --out data/history/games.csv

Each box score carries a pair of `line` records, one per team, holding runs
by inning. Their sums are the final score. A home team that led after the
top of the ninth has one fewer entry than the visitor, because it never
batted -- the same rule the simulator has to reproduce, visible directly in
the data.

Temperature, wind and park are pulled through as well; they are the inputs a
scoring environment adjustment would need.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

FIELDS = [
    "game_id", "date", "away_team", "home_team", "park",
    "away_runs", "home_runs", "innings", "day_night",
    "temp", "wind_dir", "wind_speed", "sky", "precip",
]


def parse_file(path: Path) -> list[dict]:
    """Read one Retrosheet box score file into game records."""
    games: list[dict] = []
    current: dict | None = None
    lines: dict[str, list[int]] = {}

    def flush() -> None:
        if current is None:
            return
        if current.get("gametype") != "regular":
            return
        if "0" not in lines or "1" not in lines:
            return
        away, home = lines["0"], lines["1"]
        games.append({
            "game_id": current.get("game_id", ""),
            "date": current.get("date", ""),
            "away_team": current.get("visteam", ""),
            "home_team": current.get("hometeam", ""),
            "park": current.get("site", ""),
            "away_runs": sum(away),
            "home_runs": sum(home),
            # The visitor always bats every inning, so its line length is the
            # game's length.
            "innings": len(away),
            "day_night": current.get("daynight", ""),
            "temp": current.get("temp", ""),
            "wind_dir": current.get("winddir", ""),
            "wind_speed": current.get("windspeed", ""),
            "sky": current.get("sky", ""),
            "precip": current.get("precip", ""),
        })

    with open(path, "r", encoding="latin-1") as handle:
        for raw in handle:
            parts = raw.rstrip("\n").split(",")
            if not parts:
                continue
            tag = parts[0]
            if tag == "id":
                flush()
                current = {"game_id": parts[1]}
                lines = {}
            elif tag == "info" and current is not None and len(parts) >= 3:
                current[parts[1]] = parts[2]
            elif tag == "line" and current is not None and len(parts) >= 3:
                side = parts[1]
                try:
                    lines[side] = [int(x) for x in parts[2:] if x != ""]
                except ValueError:
                    lines[side] = []
    flush()
    return games


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrosheet", required=True,
                        help="path to a retrosheet checkout")
    parser.add_argument("--seasons", nargs="+", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.retrosheet) / "seasons"
    rows: list[dict] = []
    for season in args.seasons:
        season_dir = root / str(season)
        found = sorted(season_dir.glob("*.EB*"))
        if not found:
            print(f"  {season}: no box score files")
            continue
        before = len(rows)
        for path in found:
            rows.extend(parse_file(path))
        print(f"  {season}: {len(rows) - before} regular season games")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} games to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
