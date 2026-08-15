"""Traditional Chinese team names, as the Asian boards print them.

The odds feed and the board JSON both key on the English names, so those stay
the identifiers. This is display only: reports read against a Chinese-language
board are far easier to check line by line when the names match the screen.

`zh` passes anything unrecognised straight through rather than raising, so a
relocation or a new franchise degrades to an English name in one column
instead of killing a pricing run.
"""

from __future__ import annotations

CHINESE_NAMES: dict[str, str] = {
    "Arizona Diamondbacks": "響尾蛇",
    "Athletics": "運動家",
    "Atlanta Braves": "勇士",
    "Baltimore Orioles": "金鶯",
    "Boston Red Sox": "紅襪",
    "Chicago Cubs": "小熊",
    "Chicago White Sox": "白襪",
    "Cincinnati Reds": "紅人",
    "Cleveland Guardians": "守護者",
    "Colorado Rockies": "落磯山",
    "Detroit Tigers": "老虎",
    "Houston Astros": "太空人",
    "Kansas City Royals": "皇家",
    "Los Angeles Angels": "天使",
    "Los Angeles Dodgers": "道奇",
    "Miami Marlins": "馬林魚",
    "Milwaukee Brewers": "釀酒人",
    "Minnesota Twins": "雙城",
    "New York Mets": "大都會",
    "New York Yankees": "洋基",
    "Philadelphia Phillies": "費城人",
    "Pittsburgh Pirates": "海盜",
    "San Diego Padres": "教士",
    "San Francisco Giants": "巨人",
    "Seattle Mariners": "水手",
    "St. Louis Cardinals": "紅雀",
    "Tampa Bay Rays": "光芒",
    "Texas Rangers": "遊騎兵",
    "Toronto Blue Jays": "藍鳥",
    "Washington Nationals": "國民",
}


def zh(team: str) -> str:
    """Chinese name for a team, or the input unchanged if unknown."""
    return CHINESE_NAMES.get(team, team)


def zh_matchup(away: str, home: str) -> str:
    return f"{zh(away)} @ {zh(home)}"


def width(text: str) -> int:
    """Display width, counting CJK characters as two columns.

    Python pads by character count, so a column holding Chinese names comes
    out ragged unless the padding is computed from display width instead.
    """
    return sum(2 if ord(ch) > 0x2E7F else 1 for ch in text)


def pad(text: str, columns: int) -> str:
    """Left-align `text` in a field `columns` display-columns wide."""
    return text + " " * max(0, columns - width(text))
