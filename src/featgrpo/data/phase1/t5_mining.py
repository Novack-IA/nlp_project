"""A-P5: mineração adversarial de pedidos benignos que o Gemma-3-4B recusa.

Fluxo em 3 passos, trocando arquivos entre os jobs (o 4B roda num job, gerador e juiz no outro):
  1. variants --gen-url U        : 2 reescritas "alarmantes" de cada Q⁰ (mesma intenção benigna)
                                  → PHASE1_DATA/mining_variants.jsonl
  2. t4_behavior generate --set mining --url <4B>   (job de modelo)
  3. confirm --qg-url --wg-url --judge-url : rotula as respostas do 4B (t4_behavior label --set mining)
     e, nas recusadas, confirma que o pedido continua benigno e é o mesmo pedido do Q⁰ original
                                  → datasets/phase1/hard_benign_mined.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from ..phase0 import guards as G
from ..phase0.common import LLM, extract_json, get_cache, no_thinking, pmap, read_jsonl, write_json, write_jsonl
from ..phase0.p4_label import consolidate_intent
from .common1 import (ACTIVE_GUARDS, GEN_MODEL, JUDGE_MODEL, PAIRS, chat_template, data1, fill, judge_json, log,
                      out_dir)

N_VAR = 2


def variants(a):
    out = data1() / "mining_variants.jsonl"
    if out.exists():
        log(f"{out} já existe"); return
    gen = LLM(a.gen_url, GEN_MODEL); gen.wait()
    cache = get_cache()
    system, utmpl = chat_template("mining_v1")
    pairs = read_jsonl(PAIRS)

    def run(p):
        user = fill(utmpl, QB=p["q_border"], TRIGGER=p["trigger"])
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        rows = []
        for k in range(N_VAR):
            r = cache.call(GEN_MODEL, "p1:mining_v1", {"user": user, "seed": k},
                           lambda: gen.chat(msgs, temperature=0.9, top_p=0.95, seed=k, max_tokens=200,
                                            extra=no_thinking(GEN_MODEL))[0])
            try:
                t = str(extract_json(r["text"])["prompt"]).strip()
            except (ValueError, KeyError, TypeError):
                continue
            if t:
                rows.append({"item_id": f"{p['group_id']}:mine{k}", "group_id": p["group_id"], "text": t,
                             "orig": p["q_border"], "trigger": p["trigger"], "domain": p["domain"], "split": p["split"]})
        return rows
    rows = [r for rs in pmap(run, pairs, workers=96, label="mineração/variantes") for r in rs]
    write_jsonl(out, rows)
    log(f"variantes: {len(rows)}")


def confirm(a):
    d = data1()
    beh = {r["item_id"]: r for r in read_jsonl(d / "behavior_mining.jsonl")}
    refused = [r for r in beh.values() if r["labels"]["final"] in ("refusal", "partial_refusal")]
    log(f"mineração: {len(refused)}/{len(beh)} variantes recusadas pelo 4B")
    cache = get_cache()
    qg = LLM(a.qg_url, G.QG_MODEL); wg = LLM(a.wg_url, G.WG_MODEL); jl = LLM(a.judge_url, JUDGE_MODEL)
    for s in (qg, wg, jl):
        s.wait()
    _, utmpl = chat_template("cell_judge_v1")
    schema = {"type": "object", "properties": {
        "requests": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "object", "properties": {
            "id": {"enum": ["R1", "R2"]}, "intent": {"enum": ["harmful", "benign"]}, "natural": {"type": "boolean"}},
            "required": ["id", "intent", "natural"]}},
        "pairs": {"type": "array", "minItems": 1, "maxItems": 1, "items": {"type": "object", "properties": {
            "pair": {"enum": ["P1"]}, "same_request": {"type": "boolean"}}, "required": ["pair", "same_request"]}}},
        "required": ["requests", "pairs"]}

    def run(r):
        user = fill(utmpl, REQUESTS=json.dumps([{"id": "R1", "text": r["text"]}, {"id": "R2", "text": r["orig"]}],
                                               ensure_ascii=False, indent=1),
                    PAIRS=json.dumps([{"pair": "P1", "requests": ["R1", "R2"]}]))
        j = judge_json(jl, cache, "cell_judge_v1", user, schema, lambda o: len(o["requests"]) == 2)
        ji = {x["id"]: x for x in j["requests"]} if j else {}
        g = G.run_guards(qg, wg, cache, r["text"], None)
        intent = consolidate_intent(g, ji.get("R1", {}).get("intent"), ACTIVE_GUARDS, "majority")
        same = bool(j and j["pairs"][0]["same_request"])
        natural = ji.get("R1", {}).get("natural") is not False
        return {**r, "intent": intent, "same_as_orig": same, "natural": natural,
                "keep": intent == "benign" and same and natural}
    rows = pmap(run, refused, workers=48, label="mineração/confirmação")
    keep = [{k: r[k] for k in ("item_id", "group_id", "text", "orig", "trigger", "domain", "split", "response")}
            | {"class_4b": r["labels"]["final"]} for r in rows if r["keep"]]
    write_jsonl(out_dir() / "hard_benign_mined.jsonl", keep)
    rep = {"variants": len(beh), "refused_by_4b": len(refused), "kept": len(keep),
           "reasons": dict(Counter("ok" if r["keep"] else f"intent={r['intent']},same={r['same_as_orig']}"
                                   for r in rows)),
           "by_domain": dict(Counter(r["domain"] for r in keep))}
    write_json(d / "mining_report.json", rep)
    log(f"mineração: {rep}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["variants", "confirm"])
    for s in ("gen", "qg", "wg", "judge"):
        ap.add_argument(f"--{s}-url")
    a = ap.parse_args()
    variants(a) if a.cmd == "variants" else confirm(a)


if __name__ == "__main__":
    main()
