"""Compara duas rodadas da Fase 0 (ex.: v5 × v5b) para medir o custo de qualidade da expansão.

    python -m featgrpo.data.phase0.compare_versions data/phase0_b200 data/phase0_v5b [--out arquivo.md]

Métricas (sobre os grupos do export final de cada versão, salvo indicação):
  - funil e taxa de validade por domínio;
  - força do rótulo: fração dos itens que decidem a validade (intenções de Q⁺/Q⁰ e respostas
    r_unsafe, r_safe_A, r_safe_helpful(Q⁰), r_refusal(Q⁰)) com consenso 3/3 (juiz + 2 guards)
    em vez de só maioria 2/3;
  - taxa de `disputed` no P4; Q⁰ por retrieval × geração; cos(Q⁺, Q⁰) do retrieval;
  - hardness, HCR/ORR de base (r_natural), severidade PKU, sub-buckets de bias;
  - na versão expandida: sementes antigas × novas (a expansão piora a qualidade?).
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter
from pathlib import Path

from . import guards as G
from .common import REPO, read_jsonl
from .p4_label import _compatible

CONTRAST = {("harm", "unsafe"), ("harm", "safe_A"), ("border", "safe_helpful"), ("border", "refusal")}
GUARDS = ["wildguard", "qwen3guard"]


def _dir(p: str) -> Path:
    return Path(p) if Path(p).is_absolute() else REPO / p


def load(d: Path) -> dict:
    groups = read_jsonl(d / "groups_p6.jsonl")
    final = [json.loads(l) for sp in ("train", "val", "test") for l in open(d / "final" / f"{sp}.jsonl")]
    fids = {g["group_id"] for g in final}
    return {"dir": d, "groups": groups, "final_ids": fids, "final": [g for g in groups if g["group_id"] in fids],
            "seeds": read_jsonl(d / "seeds.jsonl")}


def unanimous_item(r: dict) -> bool | None:
    lab = r["labels"]
    j = lab.get("judge")
    if not j or lab["final"] in ("disputed", "incoherent"):
        return None
    return all(_compatible(n, lab.get(n), j["class"], r["q_role"] == "harm") for n in GUARDS)


def unanimous_intent(labels: dict) -> bool | None:
    if labels["final"] == "disputed" or labels.get("judge") is None:
        return None
    return all(G.g_intent(n, labels.get(n)) == labels["final"] for n in GUARDS) and labels["judge"] == labels["final"]


def strength(groups: list[dict]) -> dict:
    u = n = 0
    for g in groups:
        flags = [unanimous_intent(g["q_harm_labels"]), unanimous_intent(g["q_border_labels"])]
        flags += [unanimous_item(r) for r in g["responses"] if (r["q_role"], r["role"]) in CONTRAST]
        for f in flags:
            if f is not None:
                n += 1
                u += f
    return {"unanimous_share": u / n if n else None, "n_items": n}


def nat(g, q):
    return next(r for r in g["responses"] if r["role"] == "natural" and r["q_role"] == q)["labels"]["final"]


def summarize(gs: list[dict]) -> dict:
    if not gs:
        return {}
    qb = [g["q_border"] for g in gs]
    cos = [q["cos_to_q_harm"] for q in qb if q.get("cos_to_q_harm") is not None]
    return {
        "n": len(gs),
        "by_domain": dict(Counter(g["domain"] for g in gs)),
        "retrieved_share": sum(q["method"] == "retrieved" for q in qb) / len(gs),
        "cos_retrieved_mean": round(st.mean(cos), 3) if cos else None,
        "hardness_gt0_share": sum((q.get("hardness") or 0) > 0 for q in qb) / len(gs),
        "hcr": sum(nat(g, "harm") == "unsafe_compliance" for g in gs) / len(gs),
        "orr": sum(nat(g, "border") in ("refusal", "partial_refusal") for g in gs) / len(gs),
        "severity": dict(sorted(Counter(g["severity"] for g in gs).items())),
        "bias_subbuckets": dict(Counter(g.get("subdomain_judge") or g.get("subdomain") for g in gs if g["domain"] == "bias")),
        "label_strength": strength(gs),
    }


def disputed_rate(groups):
    items = [r["labels"]["final"] for g in groups for r in g["responses"]]
    return sum(x == "disputed" for x in items) / len(items)


def valid_rate(groups):
    out = {}
    for dom in sorted({g["domain"] for g in groups}):
        gs = [g for g in groups if g["domain"] == dom]
        out[dom] = f"{sum(g['valid'] for g in gs)}/{len(gs)} ({sum(g['valid'] for g in gs) / len(gs):.0%})"
    return out


def fmt(x):
    if isinstance(x, float):
        return f"{x:.3f}"
    if isinstance(x, dict):
        return ", ".join(f"{k}: {fmt(v)}" for k, v in x.items())
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--out")
    a = ap.parse_args()
    A, B = load(_dir(a.a)), load(_dir(a.b))
    base_ids = {s["seed_id"] for s in A["seeds"]}
    old_b = [g for g in B["groups"] if g["seed_id"] in base_ids]
    new_b = [g for g in B["groups"] if g["seed_id"] not in base_ids]
    sa, sb = summarize(A["final"]), summarize(B["final"])
    sb_old_final = summarize([g for g in B["final"] if g["seed_id"] in base_ids])
    sb_new_final = summarize([g for g in B["final"] if g["seed_id"] not in base_ids])
    va = {g["group_id"] for g in A["groups"] if g["valid"]}
    vb_old = {g["group_id"] for g in old_b if g["valid"]}

    L = [f"# Comparação `{A['dir'].name}` (A) × `{B['dir'].name}` (B)", ""]
    L += ["## Tamanho", "", "| | A | B |", "|---|---|---|",
          f"| sementes (P1) | {len(A['seeds'])} | {len(B['seeds'])} |",
          f"| grupos (P2) | {len(A['groups'])} | {len(B['groups'])} |",
          f"| válidos (P5) | {sum(g['valid'] for g in A['groups'])} | {sum(g['valid'] for g in B['groups'])} |",
          f"| export final (P8) | {sa['n']} | {sb['n']} |",
          f"| final por domínio | {fmt(sa['by_domain'])} | {fmt(sb['by_domain'])} |", ""]
    L += ["## Taxa de validade por domínio (P5)", "", "| domínio | A | B (todas) | B: sementes antigas | B: sementes novas |",
          "|---|---|---|---|---|"]
    for dom in sorted(valid_rate(A["groups"])):
        L.append(f"| {dom} | {valid_rate(A['groups'])[dom]} | {valid_rate(B['groups']).get(dom)} | "
                 f"{valid_rate(old_b).get(dom, '-')} | {valid_rate(new_b).get(dom, '-') if new_b else '-'} |")
    L += ["", f"Estabilidade dos grupos antigos: {len(va & vb_old)} válidos em A e em B; {len(va - vb_old)} só em A; "
          f"{len(vb_old - va)} só em B (variação de rótulo entre rodadas).", ""]
    L += ["## Qualidade do export final", "",
          "| métrica | A | B | B: antigas | B: novas |", "|---|---|---|---|---|"]
    rows = [("força do rótulo: itens com consenso 3/3", lambda s: s["label_strength"]["unanimous_share"]),
            ("Q⁰ por retrieval (prompts reais)", lambda s: s["retrieved_share"]),
            ("cos médio Q⁺–Q⁰ (retrieval)", lambda s: s["cos_retrieved_mean"]),
            ("Q⁰ com hardness > 0", lambda s: s["hardness_gt0_share"]),
            ("HCR base (r_natural, Q⁺)", lambda s: s["hcr"]),
            ("ORR base (r_natural, Q⁰)", lambda s: s["orr"]),
            ("severidade PKU", lambda s: s["severity"]),
            ("sub-buckets bias", lambda s: s["bias_subbuckets"])]
    for name, f in rows:
        L.append(f"| {name} | {fmt(f(sa))} | {fmt(f(sb))} | {fmt(f(sb_old_final)) if sb_old_final else '-'} | "
                 f"{fmt(f(sb_new_final)) if sb_new_final else '-'} |")
    L += ["", f"Taxa de `disputed` nas respostas (P4, todos os grupos): A = {disputed_rate(A['groups']):.3f}; "
          f"B = {disputed_rate(B['groups']):.3f}; B antigas = {disputed_rate(old_b):.3f}; "
          f"B novas = {disputed_rate(new_b) if new_b else float('nan'):.3f}", ""]
    L += [f"Sobreposição dos exports finais: {len(A['final_ids'] & B['final_ids'])} grupos de A estão no final de B.", ""]
    text = "\n".join(L) + "\n"
    out = Path(a.out) if a.out else _dir(a.b) / f"compare_{A['dir'].name}_vs_{B['dir'].name}.md"
    out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
