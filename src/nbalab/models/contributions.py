"""Which parts of the game context move the projection, and by how much.

The projection is compared with a *neutral* game for the same player: a
league-average opponent, half home and half away, one day of rest, his
recent average minutes, no named defender, and form equal to his longer-run
level. Each context group (opponent, home/away, rest, ...) is then switched
from neutral to its real value.

The order of switching matters when effects interact (a back-to-back costs
more minutes for a starter), so each group gets its **Shapley value**: its
average marginal effect over every possible order. With g groups that is 2^g
evaluations of the model, cheap for g <= 7. Shapley values always add up
exactly to projection - neutral projection.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable, Iterable

Coalition = frozenset[str]


@dataclass(frozen=True)
class Contribution:
    """One bar of the "what moves the projection" chart."""

    label: str
    value: float
    group: str
    is_proxy: bool = False


def all_coalitions(groups: Iterable[str]) -> list[Coalition]:
    g = list(groups)
    return [frozenset(c) for r in range(len(g) + 1) for c in itertools.combinations(g, r)]


def shapley_values(groups: list[str], values: dict[Coalition, float]) -> dict[str, float]:
    """Exact Shapley values from the value of every coalition (set of groups switched on).

    phi_i = sum over S not containing i of |S|! (n - |S| - 1)! / n! x (v(S + i) - v(S)).
    """
    n = len(groups)
    out = {}
    for g in groups:
        total = 0.0
        for s in all_coalitions([x for x in groups if x != g]):
            weight = math.factorial(len(s)) * math.factorial(n - len(s) - 1) / math.factorial(n)
            total += weight * (values[s | {g}] - values[s])
        out[g] = total
    return out


def shapley(groups: list[str], value_fn: Callable[[Coalition], float]) -> dict[str, float]:
    """Convenience wrapper that evaluates ``value_fn`` on every coalition."""
    return shapley_values(groups, {s: value_fn(s) for s in all_coalitions(groups)})
