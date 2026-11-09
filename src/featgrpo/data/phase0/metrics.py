"""Métricas de acordo: Cohen κ e Krippendorff α (nominal)."""
from __future__ import annotations

from collections import Counter
from itertools import permutations


def cohen_kappa(a: list, b: list) -> float:
    assert len(a) == len(b) and a, "listas vazias ou de tamanhos diferentes"
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def krippendorff_alpha_nominal(units: list[list]) -> float:
    """units: uma lista de valores por unidade (None = sem anotação)."""
    o: Counter = Counter()
    for u in units:
        vals = [v for v in u if v is not None]
        m = len(vals)
        if m < 2:
            continue
        for c, k in permutations(range(m), 2):
            o[(vals[c], vals[k])] += 1 / (m - 1)
    n_c: Counter = Counter()
    for (c, _k), w in o.items():
        n_c[c] += w
    n = sum(n_c.values())
    if n <= 1:
        return float("nan")
    do = sum(w for (c, k), w in o.items() if c != k)
    de = sum(n_c[c] * n_c[k] for c in n_c for k in n_c if c != k) / (n - 1)
    if de == 0:
        return 1.0
    return 1 - do / de
