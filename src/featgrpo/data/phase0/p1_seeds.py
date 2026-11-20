"""P1 — sementes Q⁺ do PKU-SafeRLHF (§P1).

    python -m featgrpo.data.phase0.p1_seeds --embed-url URL --judge-url URL [--per-domain N]
"""
from __future__ import annotations

import argparse
import random
from collections import Counter

from .common import LLM, data_dir, get_cache, load_config, log, normalize, pmap, raw_dir, read_jsonl, sha256, write_json, write_jsonl


def domain_of(harm_category: dict, cfg: dict) -> str | None:
    for d in cfg["domain_priority"]:
        if any(harm_category.get(c) for c in cfg["domains"][d]["pku_categories"]):
            return d
    return None


def _b(v) -> bool:
    return v if isinstance(v, bool) else str(v).lower() == "true"


def is_seed(r: dict, cfg: dict) -> bool:
    s0, s1 = _b(r["is_response_0_safe"]), _b(r["is_response_1_safe"])
    if s0 == s1:
        return False                                     # precisa de 1 safe + 1 unsafe
    u = 0 if not s0 else 1                               # índice da resposta unsafe
    if int(r[f"response_{u}_severity_level"]) < cfg["seeds"]["min_severity"]:
        return False
    if cfg["seeds"]["require_same_generator"] and r["response_0_source"] != r["response_1_source"]:
        return False                                     # controle de estilo
    return domain_of(r[f"response_{u}_harm_category"], cfg) is not None


def to_seed(r: dict, cfg: dict) -> dict:
    u = 0 if not _b(r["is_response_0_safe"]) else 1
    s = 1 - u
    hc = r[f"response_{u}_harm_category"]
    return {"q_harm": r["prompt"], "r_unsafe": r[f"response_{u}"], "r_safe_A": r[f"response_{s}"],
            "generator": r[f"response_{u}_source"], "domain": domain_of(hc, cfg), "subdomain": None,
            "severity": int(r[f"response_{u}_severity_level"]),
            "harm_categories": sorted(k for k, v in hc.items() if v),
            "pku_row_sha256": [r.get("response_0_sha256"), r.get("response_1_sha256")]}


def dedup(rows: list[dict], priority: list[str]) -> list[dict]:
    """Uma linha por prompt normalizado; gerador mais recente primeiro."""
    rank = {g: i for i, g in enumerate(priority)}
    best: dict[str, dict] = {}
    for r in rows:
        k = normalize(r["q_harm"])
        if k not in best or rank.get(r["generator"], 99) < rank.get(best[k]["generator"], 99):
            best[k] = r
    return list(best.values())


def sample_domain(pool: list[dict], n: int, rng: random.Random, quotas: dict[str, int]) -> list[dict]:
    pool = sorted(pool, key=lambda s: s["seed_id"])
    out = []
    for sb, q in quotas.items():
        items = [s for s in pool if s.get("subdomain") == sb]
        out += rng.sample(items, min(q, len(items)))
    rest = [s for s in pool if s.get("subdomain") not in quotas]
    out += rng.sample(rest, max(0, min(n - len(out), len(rest))))
    return out


def classify_subbuckets(cands: list[dict], judge, batch: int = 25) -> None:
    chunks = [cands[i:i + batch] for i in range(0, len(cands), batch)]

    def run(ch):
        items = [{"id": s["seed_id"], "request": s["q_harm"]} for s in ch]
        return judge.label_items("subbucket_v1", items, "bias_subbucket", ["lgbt", "race", "other"])

    for ch, res in zip(chunks, pmap(run, chunks, workers=16, label="P1 sub-bucket")):
        for s in ch:
            s["subdomain"] = res.get(s["seed_id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-url", required=True)
    ap.add_argument("--judge-url", required=True)
    ap.add_argument("--per-domain", type=int, default=None)
    ap.add_argument("--extend-from", default=None,
                    help="seeds.jsonl de uma rodada anterior: mantém essas sementes (mesma ordem) e "
                         "completa cada domínio até candidates_per_domain com sementes novas (seed+1)")
    a = ap.parse_args()
    cfg = load_config()
    d = data_dir()
    from .judge import Judge
    from .p0_download import Exclusion, embed_texts

    pku = read_jsonl(raw_dir() / "pku.jsonl")
    seeds = [to_seed(r, cfg) for r in pku if is_seed(r, cfg)]
    n_filter = len(seeds)
    seeds = dedup(seeds, cfg["seeds"]["generator_priority"])
    n_dedup = len(seeds)
    for s in seeds:
        s["seed_id"] = "s_" + sha256(normalize(s["q_harm"]))[:12]

    emb = LLM(a.embed_url, cfg["models"]["embedder"])
    emb.wait()
    E = Exclusion(d / "exclusion", cfg["exclusion"]["cos_threshold"])
    qe = embed_texts(emb, [s["q_harm"] for s in seeds], "P1 Q⁺")
    keep = ~E.mask([s["q_harm"] for s in seeds], qe)
    seeds = [s for s, k in zip(seeds, keep) if k]

    judge_llm = LLM(a.judge_url, cfg["models"]["judge"]["model"])
    judge_llm.wait()
    judge = Judge(judge_llm, get_cache(), cfg)
    bias = [s for s in seeds if s["domain"] == "bias"]
    classify_subbuckets(bias, judge)
    for s in bias:
        s["subdomain"] = s["subdomain"] or "other"

    n = a.per_domain or cfg["candidates_per_domain"]
    dc = cfg["domains"]["bias"]
    sub_quota = {k: max(1, round(v * dc["subbucket_oversample"] * n / cfg["candidates_per_domain"]))
                 for k, v in dc["subbucket_target"].items()}
    out, deficit = [], {}
    if a.extend_from:                      # §12: dobra candidatos mantendo as sementes já usadas
        base = read_jsonl(a.extend_from)
        used = {s["seed_id"] for s in base}
        by_id = {s["seed_id"]: s for s in seeds}
        missing = [s["seed_id"] for s in base if s["seed_id"] not in by_id]
        assert not missing, f"sementes da base fora do pool atual: {missing[:5]}"
        out = list(base)
        rng = random.Random(cfg["seed"] + 1)
        for dom in cfg["domain_priority"]:
            have = [s for s in base if s["domain"] == dom]
            pool = [s for s in seeds if s["domain"] == dom and s["seed_id"] not in used]
            quotas = {}
            if dom == "bias":
                quotas = {k: max(0, q - sum(s["subdomain"] == k for s in have)) for k, q in sub_quota.items()}
            got = sample_domain(pool, n - len(have), rng, quotas)
            if len(have) + len(got) < n:
                deficit[dom] = n - len(have) - len(got)
            out += got
        log(f"P1 extensão: base {len(base)} + novas {len(out) - len(base)}")
    else:
        rng = random.Random(cfg["seed"])
        for dom in cfg["domain_priority"]:
            pool = [s for s in seeds if s["domain"] == dom]
            got = sample_domain(pool, n, rng, sub_quota if dom == "bias" else {})
            if len(got) < n:
                deficit[dom] = n - len(got)
            out += got
    write_jsonl(d / "seeds.jsonl", out)
    rep = {"after_filter": n_filter, "after_dedup": n_dedup, "after_exclusion": len(seeds),
           "eligible_per_domain": dict(Counter(s["domain"] for s in seeds)),
           "bias_subbucket_pool": dict(Counter(s["subdomain"] for s in bias)),
           "sampled_per_domain": dict(Counter(s["domain"] for s in out)),
           "sampled_bias_subbuckets": dict(Counter(s["subdomain"] for s in out if s["domain"] == "bias")),
           "deficit": deficit, "extended_from": a.extend_from, "judge": judge_llm.stats}
    write_json(d / "p1_report.json", rep)
    log(f"P1 ok {rep}")


if __name__ == "__main__":
    main()
