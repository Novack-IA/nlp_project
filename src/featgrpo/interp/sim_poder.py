"""Poder da regra de classes da frente C, sem dados de ativação (base da seção 4 do pré-registro).

Duas partes:
1. Contagens no texto do dataset: grupos por split e domínio hoje, e projeções do 2×2 em cenários de
   rendimento (fração dos 2.428 pares que vira grupo completo).
2. Simulação: features "plantadas" com efeito conhecido geram ativações nas 4 células de n grupos
   (esparsidade tipo ReLU) e passam pela regra de S_intent do pré-registro:
   |d_I| >= 0,5, p_I < corte (aproxima q < 0,01 do BH) e IC 95% de d_L (bootstrap) dentro de (-0,2; 0,2),
   com a feature ativa em >= 5% dos itens de alguma célula.

    PYTHONPATH=src python -m featgrpo.interp.sim_poder            # grava results/frente-c/poder.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
from scipy.stats import wilcoxon

from featgrpo.data.phase1.validate_template import normalize_trigger
from featgrpo.interp.data import REPO, load_pairs

OUT = REPO / "results" / "frente-c" / "poder.json"


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return None


# ---------- 1. contagens ----------

def counts(survival=(0.20, 0.33, 0.50), loss_fim_pedido: float = 0.20) -> dict:
    pairs = load_pairs()
    train_trigs = {normalize_trigger(p["trigger"]) for p in pairs if p["split"] == "train"}
    by_split = Counter(p["split"] for p in pairs)
    by_dom = defaultdict(Counter)
    unseen = Counter()
    for p in pairs:
        by_dom[p["split"]][p["domain"]] += 1
        if p["split"] == "test" and normalize_trigger(p["trigger"]) not in train_trigs:
            unseen[p["domain"]] += 1
    descoberta = by_split["train"] + by_split["val"]
    desc_dom = {d: by_dom["train"][d] + by_dom["val"][d] for d in sorted(by_dom["train"])}
    cenarios = []
    for s in survival:
        cenarios.append({
            "fracao_pares_que_viram_grupo": s,
            "grupos_total": round(len(pairs) * s),
            "descoberta": round(descoberta * s),
            "descoberta_fim_pedido_apos_perda": round(descoberta * s * (1 - loss_fim_pedido)),
            "confirmacao_gatilho_inedito": round(sum(unseen.values()) * s),
            "menor_dominio_descoberta": round(min(desc_dom.values()) * s),
        })
    return {
        "pares_hoje": {"descoberta": descoberta, "confirmacao": by_split["test"],
                       "confirmacao_gatilho_inedito": sum(unseen.values()),
                       "descoberta_por_dominio": desc_dom,
                       "confirmacao_inedito_por_dominio": dict(sorted(unseen.items()))},
        "perda_suposta_fim_pedido": loss_fim_pedido,
        "cenarios_2x2": cenarios,
        "nota": "Supõe que o rendimento do 2×2 é igual em todos os splits e domínios.",
    }


# ---------- 2. simulação ----------

class Sim:
    def __init__(self, seed: int, base: float, noise: float, p_cut: float, n_boot: int):
        self.rng = np.random.default_rng(seed)
        self.base, self.noise, self.p_cut, self.n_boot = base, noise, p_cut, n_boot

    def cells(self, n, b_i, b_l, frac=1.0):
        rng = self.rng
        b = rng.normal(self.base, 1.0, n)
        on = (rng.random(n) < frac).astype(float)
        a = {(h, t): np.maximum(b + b_i * h * on + b_l * t + rng.normal(0, self.noise, n), 0.0)
             for h in (1, 0) for t in (1, 0)}
        i_g = 0.5 * (a[1, 1] + a[1, 0] - a[0, 1] - a[0, 0])
        l_g = 0.5 * (a[1, 1] + a[0, 1] - a[1, 0] - a[0, 0])
        active = max(float((v > 0).mean()) for v in a.values())
        return i_g, l_g, active

    @staticmethod
    def d(x):
        s = x.std(ddof=1)
        return float(x.mean() / s) if s > 0 else 0.0

    def true_d(self, b_i, b_l, frac=1.0):
        i_g, l_g, _ = self.cells(400_000, b_i, b_l, frac)
        return self.d(i_g), self.d(l_g)

    def solve(self, target, f, lo=0.0, hi=5.0):
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
        return (lo + hi) / 2

    def is_s_intent(self, i_g, l_g, active) -> bool:
        n = len(i_g)
        idx = self.rng.integers(0, n, (self.n_boot, n))
        lb = l_g[idx]
        sd = lb.std(axis=1, ddof=1)
        d_lb = np.where(sd > 0, lb.mean(axis=1) / np.where(sd > 0, sd, 1.0), 0.0)
        lo, hi = np.percentile(d_lb, [2.5, 97.5])
        p_i = wilcoxon(i_g, zero_method="pratt").pvalue if np.any(i_g != 0) else 1.0
        return active >= 0.05 and abs(self.d(i_g)) >= 0.5 and p_i < self.p_cut and lo > -0.2 and hi < 0.2


def simulate(ns, reps, seed, base, noise, p_cut, n_boot) -> dict:
    s = Sim(seed, base, noise, p_cut, n_boot)
    b_i06 = s.solve(0.6, lambda x: s.true_d(x, 0)[0])
    b_i10 = s.solve(1.0, lambda x: s.true_d(x, 0)[0])
    b_l01 = s.solve(0.1, lambda x: s.true_d(b_i06, x)[1])
    b_l08 = s.solve(0.8, lambda x: s.true_d(0, x)[1])
    b_tema = s.solve(0.55, lambda x: s.true_d(x, 0, 0.25)[0])
    tipos = {
        "intencao_pura": (b_i06, 0.0, 1.0),
        "intencao_forte": (b_i10, 0.0, 1.0),
        "intencao_vazamento_leve": (b_i06, b_l01, 1.0),
        "lexical": (0.0, b_l08, 1.0),
        "tema_um_dominio_25pct": (b_tema, 0.0, 0.25),
        "nula": (0.0, 0.0, 1.0),
    }
    out = {}
    for nome, (b_i, b_l, frac) in tipos.items():
        d_i, d_l = s.true_d(b_i, b_l, frac)
        prob = {str(n): sum(s.is_s_intent(*s.cells(n, b_i, b_l, frac)) for _ in range(reps)) / reps for n in ns}
        out[nome] = {"d_I_verdadeiro": round(d_i, 3), "d_L_verdadeiro": round(d_l, 3),
                     "fracao_grupos_com_efeito": frac, "P_classificada_S_intent": prob}
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ns", type=int, nargs="+", default=[150, 260, 400, 550, 680])
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", type=float, default=-0.3, help="deslocamento da base (esparsidade)")
    ap.add_argument("--noise", type=float, default=0.7)
    ap.add_argument("--p-cut", type=float, default=1e-6, help="aproxima q<0,01 do BH em ~131 mil testes")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    res = {
        "script": "src/featgrpo/interp/sim_poder.py",
        "args": vars(args),
        "git_commit": git_commit(),
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "contagens": counts(),
        "simulacao": simulate(args.ns, args.reps, args.seed, args.base, args.noise, args.p_cut, args.n_boot),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print(f"gravado em {args.out}")
    print(f"{'tipo':28s}" + "".join(f"{n:>7d}" for n in args.ns))
    for nome, r in res["simulacao"].items():
        print(f"{nome:28s}" + "".join(f"{r['P_classificada_S_intent'][str(n)]:7.0%}" for n in args.ns))


if __name__ == "__main__":
    main()
