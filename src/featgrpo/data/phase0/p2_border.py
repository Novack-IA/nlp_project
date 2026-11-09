"""P2 — construção de Q⁰: gatilho → retrieval no OR-Bench-80k → fallback de geração (§P2).

    python -m featgrpo.data.phase0.p2_border --gen-url URL --embed-url URL
"""
from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict

import numpy as np
from nltk.stem import PorterStemmer

from .common import (LLM, data_dir, extract_json, fill, get_cache, load_chat_template, load_config, log, no_thinking,
                     normalize, pmap, raw_dir, read_jsonl, strip_thinking, write_json, write_jsonl)

_stem = PorterStemmer()
_TOK = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def stems(text: str) -> list[str]:
    return [_stem.stem(t) for t in _TOK.findall((text or "").lower())]


def shares_trigger(trigger: str, text: str) -> bool:
    """Condição 1: o stem de cada palavra do gatilho aparece entre os stems do candidato."""
    ts = stems(trigger)
    if not ts:
        return False
    cs = set(stems(text))
    return all(t in cs for t in ts)


def trigger_valid(trigger: str | None, q_harm: str) -> bool:
    if not trigger or not isinstance(trigger, str):
        return False
    t = normalize(trigger)
    return bool(t) and re.search(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])", normalize(q_harm)) is not None


PATTERNS = {"homonym", "safe_target", "safe_context", "figurative_language", "historical_educational", "definitional"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-url", required=True)
    ap.add_argument("--embed-url", required=True)
    a = ap.parse_args()
    cfg = load_config()
    bc = cfg["border"]
    d = data_dir()
    cache = get_cache()
    from .p0_download import Exclusion, embed_texts

    seeds = read_jsonl(d / "seeds.jsonl")
    gen_model = cfg["models"]["generator"]
    gen = LLM(a.gen_url, gen_model)
    emb = LLM(a.embed_url, cfg["models"]["embedder"])
    gen.wait()
    emb.wait()
    extra = no_thinking(gen_model)

    # ---- P2.a gatilho
    sys_t, usr_t = load_chat_template(cfg, "trigger_v1")

    def get_trigger(s):
        msgs = [{"role": "system", "content": sys_t}, {"role": "user", "content": fill(usr_t, **{"<Q⁺>": s["q_harm"]})}]
        for temp in (0.0, 0.3):
            r = cache.call(gen_model, f"trigger_v1@{temp}", {"q": s["q_harm"]},
                           lambda: gen.chat(msgs, temperature=temp, max_tokens=64, seed=0, extra=extra)[0])
            try:
                t = extract_json(strip_thinking(r["text"])[0]).get("trigger")
            except (ValueError, AttributeError):
                t = None
            if trigger_valid(t, s["q_harm"]):
                return t
        return None

    trig = pmap(get_trigger, seeds, workers=64, label="P2.a gatilho")
    for s, t in zip(seeds, trig):
        s["trigger"] = t

    # ---- P2.b retrieval
    pool = [r["prompt"] for r in read_jsonl(raw_dir() / "orbench_80k.jsonl")]
    pool_emb = np.load(d / "exclusion" / "orbench80k_emb.npy")
    E = Exclusion(d / "exclusion", cfg["exclusion"]["cos_threshold"])
    pool_excl = E.mask(pool, pool_emb)
    ok_seeds = [s for s in seeds if s["trigger"]]
    q_emb = embed_texts(emb, [s["q_harm"] for s in ok_seeds], "P2 Q⁺ (query)", instruction=bc["query_instruction"])
    used: set[str] = set()
    groups, stats, per_dom = [], Counter(), defaultdict(Counter)
    stats["no_trigger"] = len(seeds) - len(ok_seeds)
    sims = q_emb @ pool_emb.T
    pending = []
    for i, s in enumerate(ok_seeds):                      # ordem determinística (ordem de seeds.jsonl)
        top = np.argsort(-sims[i])[: bc["top_k"]]
        hit = None
        for j in top:
            c = float(sims[i, j])
            if not (bc["cos_min"] <= c <= bc["cos_max"]):
                continue
            cand = pool[j]
            if pool_excl[j] or normalize(cand) in used or not shares_trigger(s["trigger"], cand):
                continue
            hit = (cand, c)
            break
        if hit:
            used.add(normalize(hit[0]))
            groups.append(s | {"q_border": {"text": hit[0], "method": "retrieved", "source": "or-bench-80k",
                                            "xstest_type": None, "cos_to_q_harm": round(hit[1], 4)}})
            stats["retrieved"] += 1
            per_dom[s["domain"]]["retrieved"] += 1
        else:
            pending.append(s)

    # ---- P2.c geração (fallback)
    gsys, gusr = load_chat_template(cfg, "border_gen_v1")

    def gen_border(s):
        msgs = [{"role": "system", "content": gsys},
                {"role": "user", "content": fill(gusr, **{"<Q⁺>": s["q_harm"], "<trigger>": s["trigger"]})}]
        r = cache.call(gen_model, "border_gen_v1", {"q": s["q_harm"], "t": s["trigger"]},
                       lambda: [x["text"] for x in gen.chat(msgs, temperature=bc["gen_temperature"], max_tokens=128,
                                                             seed=0, n=bc["gen_samples"], extra=extra)])
        out = []
        for t in r:
            try:
                o = extract_json(strip_thinking(t)[0])
                if isinstance(o, dict) and o.get("prompt"):
                    out.append(o)
            except ValueError:
                pass
        return out

    samples = pmap(gen_border, pending, workers=64, label="P2.c geração")
    flat = [(s, o) for s, outs in zip(pending, samples) for o in outs]
    cand_emb = embed_texts(emb, [o["prompt"] for _, o in flat], "P2 gerados") if flat else np.zeros((0, 1))
    excl = E.mask([o["prompt"] for _, o in flat], cand_emb) if flat else []
    k = 0
    for s, outs in zip(pending, samples):
        chosen = None
        for o in outs:
            ok = (not excl[k]) and shares_trigger(s["trigger"], o["prompt"]) and normalize(o["prompt"]) not in used
            k += 1
            if ok and chosen is None:
                chosen = o
        if chosen:
            used.add(normalize(chosen["prompt"]))
            pat = chosen.get("pattern") if chosen.get("pattern") in PATTERNS else None
            groups.append(s | {"q_border": {"text": chosen["prompt"], "method": "generated", "source": gen_model,
                                            "xstest_type": pat, "cos_to_q_harm": None}})
            stats["generated"] += 1
            per_dom[s["domain"]]["generated"] += 1
        else:
            stats["discarded"] += 1

    order = {s["seed_id"]: i for i, s in enumerate(seeds)}
    groups.sort(key=lambda g: order[g["seed_id"]])
    for i, g in enumerate(groups):
        g["group_id"] = f"g_{i:06d}"
        g["pipeline_version"] = cfg["version"]
    write_jsonl(d / "groups_p2.jsonl", groups)
    rep = {"stats": dict(stats), "per_domain": {k: dict(v) for k, v in per_dom.items()}, "seeds": len(seeds),
           "groups": len(groups), "generator": gen.stats}
    write_json(d / "p2_report.json", rep)
    log(f"P2 ok {rep}")


if __name__ == "__main__":
    main()
