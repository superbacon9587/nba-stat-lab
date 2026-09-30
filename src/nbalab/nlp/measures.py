"""Parse heights and weights written in everyday forms.

Heights come in many spellings: ``6'7``, ``6'7"``, ``6 foot 7``, ``6 ft 7 in``,
``6-7``, ``79 inches``, ``201 cm``, ``seven footer``. All become inches.

A single height ("a 6'7 defender") is turned into a *range*, because listed
heights are rounded and a filter on exactly 79.0 inches would keep very few
players. By default the range is +/- 1 inch (78 to 80), which is configurable.
"At least 6'9" / "shorter than 6'3" become one-sided ranges instead.
"""

from __future__ import annotations

import re
from typing import Literal

Comparison = Literal["about", "at_least", "at_most"]

# Physical bounds for an NBA player, used to reject numbers that are not heights.
MIN_HEIGHT_IN, MAX_HEIGHT_IN = 60.0, 96.0
MIN_WEIGHT_LB, MAX_WEIGHT_LB = 130.0, 400.0
OPEN_LOW_IN, OPEN_HIGH_IN = 60.0, 96.0
OPEN_LOW_LB, OPEN_HIGH_LB = 130.0, 400.0

_FEET = r"(?P<ft>[4-7])"
_INCH = r"(?P<inch>1[01]|[0-9])(?:\.(?P<half>5))?"
HEIGHT_PATTERNS: tuple[re.Pattern[str], ...] = (
    # 6'7, 6'7", 6’ 7”, 6' 7''
    re.compile(rf"\b{_FEET}\s*['’]\s*{_INCH}\s*(?:\"|”|''|in(?:ch(?:es)?)?\b)?"),
    # 6 foot 7, 6 feet 7 inches, 6ft7, 6 ft 7 in
    re.compile(rf"\b{_FEET}\s*(?:foot|feet|ft\.?)\s*(?:and\s+)?{_INCH}(?:\s*(?:inches|inch|in)\b)?", re.I),
    # 6-7 (only when followed by a player word, so "2020-21" and "6-7 record" stay out)
    re.compile(rf"\b{_FEET}-{_INCH}(?=\s+(?:defenders?|guards?|forwards?|centers?|wings?|players?|bigs?)\b)", re.I),
)
FEET_ONLY = re.compile(r"\b(?P<ft>[5-7]|seven|six)[\s-]*(?:foot|feet|ft|footer)s?\b(?!\s*\d)", re.I)
INCHES_ONLY = re.compile(r"\b(?P<n>\d{2}(?:\.\d)?)\s*(?:inches|inch|in\b|\")", re.I)
CENTIMETERS = re.compile(r"\b(?P<n>\d{3})\s*(?:cm|centimeters?)\b", re.I)
WEIGHT = re.compile(r"\b(?P<n>\d{3})\s*(?:lbs?|pounds?)\b", re.I)

AT_LEAST_CUES = re.compile(r"(?:at least|taller than|over|above|bigger than|heavier than|more than|\bmin(?:imum)?)\s*$", re.I)
AT_MOST_CUES = re.compile(r"(?:at most|shorter than|under|below|smaller than|lighter than|less than|\bmax(?:imum)?)\s*$", re.I)
_WORD_FEET = {"six": 6, "seven": 7}


def _in_bounds(inches: float) -> bool:
    return MIN_HEIGHT_IN <= inches <= MAX_HEIGHT_IN


def find_height(text: str) -> tuple[float, tuple[int, int]] | None:
    """First height in ``text`` as (inches, (start, end) span), or ``None``."""
    for pat in HEIGHT_PATTERNS:
        m = pat.search(text)
        if m:
            inches = int(m["ft"]) * 12 + int(m["inch"]) + (0.5 if m["half"] else 0.0)
            if _in_bounds(inches):
                return inches, m.span()
    m = FEET_ONLY.search(text)
    if m:
        ft = m["ft"].lower()
        return float(_WORD_FEET.get(ft) or int(ft)) * 12, m.span()
    m = INCHES_ONLY.search(text)
    if m and _in_bounds(float(m["n"])):
        return float(m["n"]), m.span()
    m = CENTIMETERS.search(text)
    if m and _in_bounds(float(m["n"]) / 2.54):
        return round(float(m["n"]) / 2.54, 1), m.span()
    return None


def parse_height_inches(text: str) -> float | None:
    """Height in inches from a short string such as ``"6'7"`` or ``"79 inches"``.

    A bare number is read as inches when it is a plausible height (60-96),
    and as feet when it is 5, 6 or 7.
    """
    found = find_height(text)
    if found:
        return found[0]
    dashed = re.fullmatch(r"\s*([4-7])\s*-\s*(1[01]|[0-9])\s*", text)  # "6-7" on its own
    if dashed:
        return float(int(dashed.group(1)) * 12 + int(dashed.group(2)))
    bare = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*", text)
    if bare:
        n = float(bare.group(1))
        if _in_bounds(n):
            return n
        if n in (5, 6, 7):
            return n * 12
    return None


def find_weight(text: str) -> tuple[float, tuple[int, int]] | None:
    """First weight in pounds in ``text`` as (lbs, span), or ``None``."""
    m = WEIGHT.search(text)
    if m and MIN_WEIGHT_LB <= float(m["n"]) <= MAX_WEIGHT_LB:
        return float(m["n"]), m.span()
    return None


def parse_weight_lbs(text: str) -> float | None:
    """Weight in pounds from ``"250 lbs"`` or a bare plausible number."""
    found = find_weight(text)
    if found:
        return found[0]
    bare = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*", text)
    if bare and MIN_WEIGHT_LB <= float(bare.group(1)) <= MAX_WEIGHT_LB:
        return float(bare.group(1))
    return None


def comparison_before(text: str, start: int) -> Comparison:
    """Read "taller than", "under", ... just before a measurement."""
    head = text[max(0, start - 20):start]
    head = re.sub(r"\b(?:an?|the)\s*$", "", head.rstrip(), flags=re.I)
    if AT_LEAST_CUES.search(head):
        return "at_least"
    if AT_MOST_CUES.search(head):
        return "at_most"
    return "about"


def to_range(
    value: float, comparison: Comparison, tolerance: float, low: float, high: float
) -> tuple[float, float]:
    """(min, max) for a measurement: +/- ``tolerance`` for "about", open-ended otherwise."""
    if comparison == "at_least":
        return value, high
    if comparison == "at_most":
        return low, value
    return value - tolerance, value + tolerance


def height_range(inches: float, comparison: Comparison = "about", tolerance: float = 1.0) -> tuple[float, float]:
    """``79, "about"`` -> ``(78, 80)``; ``81, "at_least"`` -> ``(81, 96)``."""
    return to_range(inches, comparison, tolerance, OPEN_LOW_IN, OPEN_HIGH_IN)


def weight_range(lbs: float, comparison: Comparison = "about", tolerance: float = 10.0) -> tuple[float, float]:
    """``250, "about"`` -> ``(240, 260)``."""
    return to_range(lbs, comparison, tolerance, OPEN_LOW_LB, OPEN_HIGH_LB)


def format_height(inches: float) -> str:
    """``79`` -> ``6'7"``."""
    whole = int(round(inches))
    return f"{whole // 12}'{whole % 12}\""
