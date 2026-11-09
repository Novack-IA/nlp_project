"""P8 — seleção final, splits por grupo, export, manifest e relatório (§P8, §11).

    python -m featgrpo.data.phase0.p8_finalize
Usa groups_p7.jsonl (pós-auditoria) se existir; senão groups_p6.jsonl e marca o export como PROVISÓRIO.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from collections import Counter, defaultdict

from .common import data_dir, load_config, log, read_jsonl, write_json, write_jsonl


def hardness_bin(h: float | None) -> str:
    h = h or 0.0
    if h == 0:
        return "0"
    return "(0,0.5]" if h <= 0.5 else "(0.5,1]"


def split(groups: list[dict], cfg: dict) -> None:
    """Split por grupo, estratificado por domínio × hardness_bin. In-place: g['split']."""
    ratios = cfg["split"]["ratios"]
    rng = random.Random(cfg["seed"])
    strata = defaultdict(list)
    for g in groups:
        strata[(g["domain"], hardness_bin(g["q_border"].get("hardness")))].append(g)
    for key in sorted(strata):
        gs = sorted(strata[key], key=lambda g: g["group_id"])
        rng.shuffle(gs)
        n = len(gs)
        n_tr, n_va = round(n * ratios["train"]), round(n * ratios["val"])
        for i, g in enumerate(gs):
            g["split"] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")


def to_record(g: dict) -> dict:
    """Registro final no esquema da §8."""
    qb = dict(g["q_border"])
    qb.pop("ref_responses", None)
    qb["labels"] = g["q_border_labels"]
    return {"group_id": g["group_id"], "pipeline_version": g["pipeline_version"], "domain": g["domain"],
            "subdomain": g.get("subdomain_judge") or g.get("subdomain"), "trigger": g.get("trigger"),
            "q_harm": {"text": g["q_harm"], "source": "PKU-SafeRLHF", "pku_row_sha256": g["pku_row_sha256"],
                       "severity": g["severity"], "harm_categories": g["harm_categories"], "labels": g["q_harm_labels"]},
            "q_border": qb,
            "responses": [{k: r.get(k) for k in ("id", "q_role", "role", "generator", "system_prompt_id", "seed",
                                                 "text", "finish_reason", "labels")} for r in g["responses"]],
            "valid": g["valid"], "invalid_reason": g["invalid_reason"], "audited": g.get("audited", False),
            "split": g["split"]}


def _load(d, name):
    p = d / name
    return json.loads(p.read_text()) if p.exists() else None


def report(d, cfg, groups, final, provisional) -> str:
    rep = {k: _load(d, f) for k, f in [("p0", "p0_report.json"), ("p1", "p1_report.json"), ("p2", "p2_report.json"),
                                        ("p3", "p3_report.json"), ("p4", "groups_p4_report.json"),
                                        ("p5", "groups_p5_report.json"), ("p6", "groups_p6_report.json"),
                                        ("calib", "calibration.json"), ("audit", "audit/metrics.json")]}
    L = [f"# Relatório da Fase 0 — `{cfg['version']}`", "",
         f"Gerado em {time.strftime('%Y-%m-%d %H:%M')}." + (" **PROVISÓRIO: sem auditoria humana (P7).**" if provisional else ""), ""]
    L += ["## 1. Funil por etapa e domínio", "", "| domínio | elegíveis P1 | amostrados P1 | grupos P2 | válidos P5 | final P8 |",
          "|---|---|---|---|---|---|"]
    p1 = rep["p1"] or {}
    for dom in cfg["domain_priority"]:
        L.append(f"| {dom} | {p1.get('eligible_per_domain', {}).get(dom, '')} | {p1.get('sampled_per_domain', {}).get(dom, '')} | "
                 f"{sum(g['domain'] == dom for g in groups)} | {sum(g['domain'] == dom and g['valid'] for g in groups)} | "
                 f"{sum(g['domain'] == dom for g in final)} |")
    if rep["p5"]:
        L += ["", f"Motivos de invalidade (P5): `{json.dumps(rep['p5'].get('invalid_reasons'))}`"]
    L += ["", "## 2. Retrieval × geração (P2)", "", "```json", json.dumps((rep["p2"] or {}).get("per_domain"), indent=1), "```"]
    L += ["", "## 3. Calibração (κ contra XSTest)", ""]
    for name, v in ((rep["calib"] or {}).get("judges") or {}).items():
        L.append(f"- **{name}**{' (' + v['model'] + ')' if v.get('model') else ''}: κ = {v['kappa']:.3f} ({v['kind']}, n={v['n']})")
    L += ["", "## 4. Acordo entre juízes", "", f"- itens `disputed` no P4: {(rep['p4'] or {}).get('disputed_items')}",
          f"- grupos com erro do juiz: {(rep['p4'] or {}).get('judge_error_groups_per_domain')}",
          f"- consolidação: `{cfg['consolidation']['rule']}`; validade: `{cfg['validity']['required_items']}`"]
    L += ["", "## 5. Auditoria", "", "```json", json.dumps(rep["audit"], indent=1) if rep["audit"] else "pendente", "```"]
    L += ["", "## 6. HCR e ORR de base do sujeito (r_natural)", "", "| domínio | HCR (Q⁺) | ORR (Q⁰) | n |", "|---|---|---|---|"]
    for dom in cfg["domain_priority"] + ["all"]:
        gs = [g for g in groups if g["valid"] and (dom == "all" or g["domain"] == dom)]
        nat = lambda g, q: next(r for r in g["responses"] if r["role"] == "natural" and r["q_role"] == q)["labels"]["final"]
        if gs:
            hcr = sum(nat(g, "harm") == "unsafe_compliance" for g in gs) / len(gs)
            orr = sum(nat(g, "border") in ("refusal", "partial_refusal") for g in gs) / len(gs)
            L.append(f"| {dom} | {hcr:.3f} | {orr:.3f} | {len(gs)} |")
    L += ["", "## 7. Distribuição de hardness", "", "```json", json.dumps((rep["p6"] or {}).get("hardness_by_domain"), indent=1), "```"]
    L += ["", "## 8. Sub-buckets de bias", "", f"Meta: {cfg['domains']['bias']['subbucket_target']}; obtido (final): "
          f"{dict(Counter(g.get('subdomain_judge') or g.get('subdomain') for g in final if g['domain'] == 'bias'))}"]
    L += ["", "## 9. Chamadas e tokens", "", "```json",
          json.dumps({k: (rep[k] or {}).get("run") for k in ("p3", "p4", "calib")}, indent=1, default=str), "```"]
    return "\n".join(L) + "\n"


def main():
    cfg = load_config()
    d = data_dir()
    provisional = not (d / "groups_p7.jsonl").exists()
    groups = read_jsonl(d / ("groups_p6.jsonl" if provisional else "groups_p7.jsonl"))
    rng = random.Random(cfg["seed"])
    final, deficit = [], {}
    for dom in cfg["domain_priority"]:
        tgt = cfg["domains"][dom]["target"]
        valid = sorted((g for g in groups if g["valid"] and g["domain"] == dom), key=lambda g: g["group_id"])
        aud = [g for g in valid if g.get("audited")]
        rest = [g for g in valid if not g.get("audited")]
        pick = rng.sample(aud, min(tgt, len(aud)))
        pick += rng.sample(rest, min(tgt - len(pick), len(rest)))
        if len(pick) < tgt:
            deficit[dom] = tgt - len(pick)
        final += sorted(pick, key=lambda g: g["group_id"])
    split(final, cfg)
    out = d / "final"
    out.mkdir(exist_ok=True)
    hashes = {}
    for sp in ("train", "val", "test"):
        write_jsonl(out / f"{sp}.jsonl", [to_record(g) for g in final if g["split"] == sp])
        hashes[f"{sp}.jsonl"] = hashlib.sha256((out / f"{sp}.jsonl").read_bytes()).hexdigest()
    man = {"version": cfg["version"], "provisional": provisional, "seed": cfg["seed"], "files": hashes,
           "models": cfg["models"], "consolidation": cfg["consolidation"], "validity": cfg["validity"],
           "counts": {"by_domain": dict(Counter(g["domain"] for g in final)),
                      "by_split": dict(Counter(g["split"] for g in final)),
                      "by_domain_split": {f"{k[0]}/{k[1]}": v for k, v in Counter((g["domain"], g["split"]) for g in final).items()}},
           "deficit": deficit, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    write_json(out / "manifest.json", man)
    (out / "report.md").write_text(report(d, cfg, groups, final, provisional))
    log(f"P8 ok {json.dumps(man['counts'])} provisório={provisional}")


if __name__ == "__main__":
    main()
