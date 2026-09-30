"""Team accent colors, keyed by the current franchise abbreviation.

Used only as decoration (a header stripe on the results page), never to encode
data, so two similar team colors can never be confused for two series.
Each entry lists the primary color first, then fallbacks. :func:`accent`
returns the first one that stays readable on the dark surface.
"""

from __future__ import annotations

TEAM_COLORS: dict[str, tuple[str, ...]] = {
    "ATL": ("#E03A3E", "#C1D32F"), "BOS": ("#007A33", "#BA9653"), "BKN": ("#FFFFFF", "#777D84"),
    "CHA": ("#1D1160", "#00788C"), "CHI": ("#CE1141", "#FFFFFF"), "CLE": ("#860038", "#FDBB30"),
    "DAL": ("#00538C", "#B8C4CA"), "DEN": ("#0E2240", "#FEC524"), "DET": ("#C8102E", "#1D42BA"),
    "GSW": ("#1D428A", "#FFC72C"), "HOU": ("#CE1141", "#C4CED4"), "IND": ("#002D62", "#FDBB30"),
    "LAC": ("#C8102E", "#1D428A"), "LAL": ("#552583", "#FDB927"), "MEM": ("#5D76A9", "#F5B112"),
    "MIA": ("#98002E", "#F9A01B"), "MIL": ("#00471B", "#EEE1C6"), "MIN": ("#236192", "#78BE20"),
    "NOP": ("#0C2340", "#C8102E", "#85714D"), "NYK": ("#006BB6", "#F58426"), "OKC": ("#007AC1", "#EF3B24"),
    "ORL": ("#0077C0", "#C4CED4"), "PHI": ("#006BB6", "#ED174C"), "PHX": ("#1D1160", "#E56020"),
    "POR": ("#E03A3E", "#FFFFFF"), "SAC": ("#5A2D81", "#63727A"), "SAS": ("#C4CED4", "#FFFFFF"),
    "TOR": ("#CE1141", "#A1A1A4"), "UTA": ("#002B5C", "#F9A01B"), "WAS": ("#002B5C", "#E31837"),
}
TEAM_COLORS["SAN"] = TEAM_COLORS["SAS"]  # the data abbreviates the Spurs as SAN
DEFAULT_ACCENT = "#3987e5"


def _luminance(hex_color: str) -> float:
    """WCAG relative luminance of an sRGB hex color."""
    h = hex_color.lstrip("#")
    chans = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in chans]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a: str, b: str) -> float:
    """WCAG contrast ratio between two hex colors (1 to 21)."""
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def accent(abbrev: str | None, surface: str = "#1a1a19", min_contrast: float = 3.0) -> str:
    """First team color with at least ``min_contrast`` against the surface, else the default."""
    for c in TEAM_COLORS.get((abbrev or "").upper(), ()):
        if contrast(c, surface) >= min_contrast:
            return c
    return DEFAULT_ACCENT
