"""P5 — validade do grupo + regeneração (§P5).

    python -m featgrpo.data.phase0.p5_validate --subject-url U --qg-url U --wg-url U --judge-url U
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from .common import LLM, data_dir, get_cache, load_config, log, pmap, read_jsonl, write_json, write_jsonl

# (q_role, role) → (classes aceitas, entra nos contrastes da §1?)
REQUIRED = {("harm", "unsafe"): ({"unsafe_compliance"}, True),
            ("harm", "safe_A"): ({"safe_helpful", "refusal"}, True),
            ("harm", "safe_helpful"): ({"safe_helpful"}, False),
            ("harm", "refusal"): ({"refusal"}, False),
            ("border", "safe_helpful"): ({"safe_helpful"}, True),
            ("border", "refusal"): ({"refusal"}, True)}
REGEN_ROLES = {"safe_helpful", "refusal"}


def _lab(labels: dict) -> str | None:
    return labels["human"] if labels.get("human") is not None else labels.get("final")


def failures(g: dict, required: str = "all") -> list[str]:
    out = []
    if _lab(g["q_harm_labels"]) != "harmful":
        out.append("q_harm:intent")
    if _lab(g["q_border_labels"]) != "benign":
        out.append("q_border:intent")
    if not g["q_border"].get("shares_trigger"):
        out.append("q_border:shares_trigger")
    for r in g["responses"]:
        req = REQUIRED.get((r["q_role"], r["role"]))
        if req is None or (required == "contrast" and not req[1]):
            continue
        if _lab(r["labels"]) not in req[0]:
            out.append(f"{r['id']}:{r['role']}")
    return out


def regen_eligible(g: dict, required: str = "all") -> bool:
    """Só respostas do sujeito falham, todas do mesmo papel (só safe_helpful ou só refusal)."""
    fails = failures(g, required)
    if not fails or any(f.startswith("q_") for f in fails):
        return False
    roles = {f.split(":")[1] for f in fails}
    return len(roles) == 1 and roles <= REGEN_ROLES


class _CacheOnly:
    """Sujeito fora do ar: a regeneração precisa estar no cache (pré-gerada com --pregen)."""
    stats: dict = {}

    def chat(self, *a, **k):
        raise RuntimeError("regeneração fora do cache: rode p5_validate --pregen com o sujeito no ar")


def pregen(a, cfg):
    """Fase de geração: gera de antemão as regenerações (seeds 1, 2) de todos os itens regeneráveis."""
    from .p3_responses import generate
    d = data_dir()
    req = cfg["validity"]["required_items"]
    model = cfg["models"]["subject"]
    llm = LLM(a.subject_url, model)
    llm.wait()
    cache = get_cache()
    roles = {k for k, (_c, contrast) in REQUIRED.items() if k[1] in REGEN_ROLES and (req == "all" or contrast)}
    jobs = [(q, r["system_prompt_id"], seed) for g in read_jsonl(d / "groups_p3.jsonl") for r in g["responses"]
            if (r["q_role"], r["role"]) in roles
            for q in [g["q_harm"] if r["q_role"] == "harm" else g["q_border"]["text"]]
            for seed in cfg["generation"]["regen_seeds"]]
    pmap(lambda j: generate(llm, cache, cfg, model, j[0], j[1], j[2]), jobs, workers=64, label="P5 pregen")
    log(f"P5 pregen ok {len(jobs)} {llm.stats}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pregen", action="store_true")
    ap.add_argument("--subject-url")
    for n in ("qg", "wg", "judge"):
        ap.add_argument(f"--{n}-url")
    a = ap.parse_args()
    cfg = load_config()
    if a.pregen:
        return pregen(a, cfg)
    d = data_dir()
    from .p3_responses import generate
    from .p4_label import active_guards, label_group, servers
    req = cfg["validity"]["required_items"]
    qg, wg, jl, judge, cache = servers(a, cfg)
    subj_model = cfg["models"]["subject"]
    if a.subject_url:
        subj = LLM(a.subject_url, subj_model)
        subj.wait()
    else:
        subj = _CacheOnly()
    active = active_guards()
    groups = read_jsonl(d / "groups_p4.jsonl")

    def regen(g):
        if not regen_eligible(g, req):
            return 0
        n = 0
        for seed in cfg["generation"]["regen_seeds"]:
            ids = {f.split(":")[0] for f in failures(g, req)}
            for r in g["responses"]:
                if r["id"] in ids:
                    q = g["q_harm"] if r["q_role"] == "harm" else g["q_border"]["text"]
                    try:
                        o = generate(subj, cache, cfg, subj_model, q, r["system_prompt_id"], seed)
                    except RuntimeError as e:
                        log(f"P5 {g['group_id']}: {e}")
                        return n
                    r.update(text=o["text"], finish_reason=o["finish_reason"], seed=seed, regenerated=True)
                    n += 1
            label_group(g, qg, wg, judge, cache, cfg, active, only=ids)
            if not failures(g, req):
                break
        return n

    n_regen = sum(pmap(regen, groups, workers=32, label="P5 regen"))
    reasons = Counter()
    for g in groups:
        f = failures(g, req)
        g["valid"] = not f
        g["invalid_reason"] = f or None
        for x in f:
            reasons[x if x.startswith("q_") else x.split(":")[1]] += 1
    write_jsonl(d / "groups_p5.jsonl", groups)
    valid = [g for g in groups if g["valid"]]
    rep = {"groups": len(groups), "valid": len(valid), "valid_per_domain": dict(Counter(g["domain"] for g in valid)),
           "total_per_domain": dict(Counter(g["domain"] for g in groups)), "regenerated_items": n_regen,
           "invalid_reasons": dict(reasons), "target_valid": 450, "required_items": req,
           "consolidation": cfg["consolidation"]["rule"]}
    write_json(d / "groups_p5_report.json", rep)
    log(f"P5 ok {json.dumps(rep)}")


if __name__ == "__main__":
    main()
